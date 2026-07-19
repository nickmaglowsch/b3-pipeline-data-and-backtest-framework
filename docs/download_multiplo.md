# CVM "Download Múltiplo" — pre-2010 fundamentals ingestion

This document records the research behind `b3_pipeline/dm_downloader.py`,
`b3_pipeline/dm_parser.py`, and `b3_pipeline/dm_main.py`, which extend the
existing CVM fundamentals pipeline (`b3_pipeline/cvm_main.py`) back to fiscal
years ~2005–2009. CVM's open-data bulk CSV portal (`dados.cvm.gov.br`, used
by `cvm_downloader.py`/`cvm_parser.py`) only has structured DFP from 2010 and
ITR from 2011; "Download Múltiplo" is CVM's older machine-readable delivery
channel, covering ITR/DFP/IAN documents received since **02/01/2006**.

## Sources consulted

- Technical manual: https://conteudo.cvm.gov.br/menu/regulados/companhias/download_multiplo/manual_tecnico.html
- Registration page: https://conteudo.cvm.gov.br/menu/regulados/companhias/download_multiplo/index.html
- Published layout spec: http://sistemas.cvm.gov.br/port/ciasabertas/Layout_Compactado.zip
  (also mirrored under https, both work; unpacked to inspect the `versao 5`–`versao 8`
  DFP/ITR/IAN layout documents — `DFP_TXT8.doc`, `DFP_DBF8.DOC`, `DFP_XML.doc` and
  ITR/IAN equivalents)
- CVM "Nota Técnica" page (background context, corroborates the manual)

## Registration (what the user must do — nothing this code can automate)

The service requires free registration before CVM issues login credentials.
There is no self-service signup form; registration is a support request:

1. Go to CVM's external support portal: **https://sistemas.cvm.gov.br/?suporte**
2. Submit a request that includes:
   - Company or individual name
   - CNPJ or CPF
   - Complete address
   - Contact person's name
   - Contact person's job title
   - Contact person's phone number
   - Contact person's email address
   - Reason for requesting access to the download service
3. CVM reviews the request and, once approved, **emails a login and password**
   to the contact email address provided.
4. Set the credentials as environment variables before running the pipeline:
   ```
   export CVM_DM_USER="<login CVM sent you>"
   export CVM_DM_PASS="<password CVM sent you>"
   ```

`dm_downloader.py` raises `CredentialsMissingError` (message includes these
exact steps) whenever `CVM_DM_USER`/`CVM_DM_PASS` are unset.

## Protocol

Single endpoint, POST, one request per (date, document type) pair — the
protocol has no date-range parameter:

```
POST https://seguro.bmfbovespa.com.br/rad/download/SolicitaDownload.asp
  txtLogin      = username
  txtSenha      = password
  txtData       = dd/mm/aaaa      (the filing/receipt date to search)
  txtHora       = 99:99           ("00:00" covers the whole day)
  txtDocumento  = TODOS | RAD | ITR | DFP | IAN | IPE | ENET
  txtAssuntoIPE = SIM | NÃO       (optional; only relevant to IPE)
```

Response is XML, `ISO-8859-1` encoded. Each `<Link>` element is **one
company's filing package for that date** (files are packaged per submission,
not batched across companies):

```xml
<DownloadMultiplo DataSolicitada="dd/mm/aaaa hh:mm" TipoDocumento="XX...X">
  <Link url="..." Documento="ITR|DFP|IAN" ccvm="1234" DataRef="dd/mm/aaaa" Situacao="Liberado|Cancelado" />
</DownloadMultiplo>
```

Error response:

```xml
<ERROS>
  <NUMERO_DO_ERRO>9999</NUMERO_DO_ERRO>
  <DESCRICAO_DO_ERRO>error message</DESCRICAO_DO_ERRO>
  <FONTE_DO_ERRO>function name</FONTE_DO_ERRO>
</ERROS>
```

Known codes: `22014` invalid date, `22015` invalid time, `22016` **no
records found** (treated as an empty result, not an error — a company simply
filed nothing that day), `22017` invalid document type, `22013` XML
generation error, `1` incorrect login.

`dm_downloader.download_range(start, end, doc_types)` walks every date in
`[start, end]` × `doc_types`, requests links, skips `Situacao="Cancelado"`,
and downloads each `<Link>`'s file into `data/dm/` (created at runtime).

## File format (CVMWIN legacy layout)

The layout zip documents, for each CVMWIN form-version (5 through 8, the
versions that were in effect across 2006–2009), **three physical delivery
formats** for the same field set: a fixed-width `TXT` variant, a `DBF`
(dBase III) variant, and an `XML` variant. The manual's own file-format
links point to the same `Layout_Compactado.zip` for both DFP/ITR/IAN and
doesn't disambiguate which physical format Download Múltiplo actually ships
without a live account to test against.

**This implementation targets the DBF variant.** Reasoning: DBF is
self-describing — field name, type, and byte-length are declared in the
file's own header — so a correct reader needs no externally sourced byte
offsets. The TXT variant's automated text extraction (no `.doc` reader was
available in this environment — `catdoc`/`antiword`/`pandoc`/`libreoffice`
are all absent, and `strings` on the raw OLE-format `.doc` recovers the same
lossy text) preserved field **names** (`CODCVM`, `DATADFP`, `CODCONTA`,
`DESCONTA`, `VALOR1/2/3`, ...) but lost the "Tamanho" (byte-width) table
column entirely, making a fixed-width parser impossible to get right without
guessing. All documented field names are ≤10 characters, consistent with
being literal DBF field names (DBF limits names to 10 usable chars), which
further supports this choice.

**This is a documented assumption, not a verified fact** — it could not be
confirmed without live credentials. `b3_pipeline/dm_parser.py` implements a
minimal pure-Python dBase III reader (`read_dbf`, stdlib `struct` only, no
new dependency) so this assumption is easy to revisit: if real downloads
turn out to be the TXT or XML variant instead, only `dm_parser.py`'s
file-loading layer needs to change — the field-name-based extraction logic
below would need no rework.

### File layout (per filing, one ZIP)

| File | Contents |
|---|---|
| `CVM.CTR` | Control/index: `CODCVM`, `DATADFP` (period end), `TIPO_DOC` (1/2 = DFP moeda-constante/legislação-societária, 3/4 = ITR idem, 5 = IAN), `CGC` (CNPJ), `RAZAO_SOC` (company name), `DATA_EMISS`/`HORA_EMISS` (protocol/emission date-time). |
| `CONFIG.001` | `ESCALA` (currency scale: `01`=unidade, `02`=mil), `ESCALA_QTD` (share-count scale, same codes), `STATUS` (categorical: apresentação/reapresentação). |
| `DFPHDR.001` / `ITRHDR.001` / `IANHDR.001` | Header/company data (redundant CNPJ/name/period fields). |
| `DFP{C}BPAE.001`, `DFP{C}BPPE.001`, `DFP{C}DERE.001` (and `ITR` equivalents) | Balance sheet assets, balance sheet liabilities+equity, income statement — **long format**: one row per `(CODCVM, DATADFP, CODCONTA)` with `DESCONTA` (account description) and `VALOR1`/`VALOR2`/`VALOR3` (values for the 2-years-back / 1-year-back / current "último exercício" period). `VALOR3` is the one we want — directly analogous to the modern bulk format's `ORDEM_EXERC == 'ÚLTIMO'` filter. The `C`-prefixed files are **consolidated** statements and are preferred (mirroring `cvm_parser.py`'s exclusive use of `*_con` files); non-consolidated ("TIPO 1" = comercial/industrial) files are the fallback for companies with no consolidated statements. Bank/insurer layouts (TIPO 2/3) are out of scope, matching the same scope limitation already present in `cvm_parser.py`. |
| `IANCAPSO.001` | "Composição do Capital Social": one row per share class (`ITEM`, `DESCRICAO` e.g. ORDINARIA/PREFERENCIAL, `QTDEACOES`). Total shares outstanding = sum of `QTDEACOES` across classes for the same `(CODCVM, DATAIAN)`, mirroring `cvm_parser.parse_fre_zip`'s ordinary+preferred summation for the modern FRE dataset. |

### Field → pipeline-column mapping

| Pipeline column | Source |
|---|---|
| `cnpj` | `CVM.CTR.CGC`, digits only |
| `period_end` | `CVM.CTR.DATADFP` (DFP/ITR) or `IANHDR`/`IANCAPSO.DATAIAN` (IAN), format `AAAAMMDD` → `YYYY-MM-DD` |
| `filing_date` | `CVM.CTR.DATA_EMISS` — the protocol/emission date, i.e. the **point-in-time** date (never `period_end`) |
| `doc_type` | `'DFP'`, `'ITR'`, or `'IAN'`, selected by only accepting `TIPO_DOC` = `2` (DFP), `4` (ITR), `5` (IAN) — the **legislação societária** (nominal-BRL, standard corporate-law) variant, explicitly excluding `1`/`3` (moeda de capacidade aquisitiva constante, an inflation-adjusted presentation) which would be inconsistent with the modern nominal-BRL dataset |
| `filing_version` | Not present as a monotonic counter in the legacy layout (`CONFIG.STATUS` is categorical: apresentação/reapresentação, not a version integer). Derived as the 1-indexed rank of `filing_date` within `(cnpj, doc_type, period_end)` — preserves the "new version ⇒ new row" semantics `cvm_storage` relies on. |
| `revenue` | `DESCONTA` containing `"RECEITA LIQUIDA"` in the income-statement file, `VALOR3` |
| `net_income` | `DESCONTA` containing `"LUCRO LIQUIDO DO EXERCICIO"` / `"PREJUIZO LIQUIDO DO EXERCICIO"` / `"RESULTADO LIQUIDO DO EXERCICIO"` (excluding per-share lines), `VALOR3` |
| `total_assets` | `DESCONTA` containing `"ATIVO TOTAL"` in the assets file, `VALOR3` |
| `equity` | `DESCONTA` containing `"PATRIMONIO LIQUIDO"` (excluding minority-interest lines) in the liabilities+equity file, `VALOR3` |
| `shares_outstanding` | `IANCAPSO.QTDEACOES` summed across share classes, scaled by `ESCALA_QTD` |
| `ebitda`, `net_debt` | **Not populated** — see below |
| `net_income_ttm` | Left `NULL`; `cvm_main.compute_net_income_ttm` fills it in a later pipeline run |

Account matching uses the `DESCONTA` (description) text rather than
`CODCONTA` (account code), because the numeric chart-of-accounts drifted
across layout versions 5–8 (this period spans the Lei 11.638/07 accounting
reform), while the account *labels* used above are stable. This mirrors the
by-account-code approach in `cvm_parser.py`, adapted for the code instability
in this era.

`net_debt` is intentionally **not computed**: the CVMWIN chart-of-accounts
does not reliably distinguish short-term vs. long-term "Empréstimos e
Financiamentos" by `DESCONTA` alone across versions 5–8, and getting a money
figure quietly wrong is worse than leaving it `NULL` (the schema allows
`NULL`). `ebitda` has no clean legacy-era proxy either (the modern pipeline
uses EBIT account `3.05` as an EBITDA proxy; no equivalently identifiable
line exists in the legacy DRE layout without real sample data to verify
against) and is likewise left `NULL`.

### Scale (currency & quantity)

The pipeline's existing convention (`cvm_storage.py`) stores all financial
values in **thousands of BRL**. `CONFIG.ESCALA` tells us the scale of the raw
`VALOR1/2/3` figures:

- `ESCALA = '01'` (unidade): raw values are whole BRL → **divide by 1000**
- `ESCALA = '02'` (mil): raw values are already thousands → no conversion

Missing `CONFIG` rows default to `'02'` (thousands — the dominant convention
for listed companies), logged as a warning. `ESCALA_QTD` scales
`IANCAPSO.QTDEACOES` the same way, but toward an absolute share count (`mil`
⇒ multiply by 1000).

## Known limitations

- **Not integration-tested against real files.** No CVM Download Múltiplo
  account was available to this implementation. `tests/test_dm_parser.py`
  verifies the parser against synthetic DBF fixtures built directly from the
  documented field names/semantics above (11 tests, all passing) — but the
  real-world file format (DBF vs. TXT vs. XML; exact `DESCONTA` label text;
  whether `CVM.CTR`/`CONFIG` cover single or multiple filings per ZIP) has
  not been verified against a real downloaded file. Once credentials are
  obtained (see Registration above), download a handful of real filings with
  `--parse-only` disabled, inspect them, and adjust `dm_parser.py`'s file
  loading if the actual format differs from the DBF assumption.
- `ebitda` and `net_debt` are always `NULL` for Download Múltiplo rows (see
  above).
- Financial institutions and insurers (CVMWIN "TIPO 2"/"TIPO 3" statement
  layouts) are not parsed — same scope limitation as the modern
  `cvm_parser.py`.

## Usage

```bash
export CVM_DM_USER="..."
export CVM_DM_PASS="..."
python -m b3_pipeline.dm_main --start 2006-01-01 --end 2010-12-31 --types ITR,DFP,IAN

# Or, to just (re-)parse files already downloaded into data/dm/:
python -m b3_pipeline.dm_main --parse-only
```

After it finishes, re-run `python -m b3_pipeline.cvm_main --skip-ticker-fetch`
to recompute `net_income_ttm` and rebuild the `fundamentals_monthly` snapshot
with the newly ingested rows (the CLI prints this reminder automatically).
