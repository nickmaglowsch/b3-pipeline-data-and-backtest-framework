"""
Client for CVM's "Download Múltiplo de Informações sobre Companhias" service.

Protocol (from https://conteudo.cvm.gov.br/menu/regulados/companhias/
download_multiplo/manual_tecnico.html):

  POST https://seguro.bmfbovespa.com.br/rad/download/SolicitaDownload.asp
    txtLogin   = username issued by CVM after registration
    txtSenha   = password issued by CVM after registration
    txtData    = dd/mm/aaaa  (the filing/receipt date to search)
    txtHora    = 99:99       (initial search time, "00:00" covers the whole day)
    txtDocumento = TODOS | RAD | ITR | DFP | IAN | IPE | ENET

  Response is XML (ISO-8859-1). On success:
    <DownloadMultiplo DataSolicitada="..." TipoDocumento="...">
      <Link url="..." Documento="ITR|DFP|IAN|..." ccvm="..." DataRef="..." Situacao="Liberado|Cancelado" />
      ...
    </DownloadMultiplo>
  On failure:
    <ERROS><NUMERO_DO_ERRO>NNNNN</NUMERO_DO_ERRO><DESCRICAO_DO_ERRO>...</DESCRICAO_DO_ERRO></ERROS>
  Notable codes: 22016 = no records found for that date/type (not fatal --
  just means nothing was filed that day); 1 = incorrect login.

Each <Link> is one company's filing package for that date (delivered
per-submission, not batched across companies) -- see docs/download_multiplo.md.

Registration (required before CVM_DM_USER/CVM_DM_PASS exist): see
REGISTRATION_INSTRUCTIONS below, or docs/download_multiplo.md.
"""
from __future__ import annotations

import logging
import os
import time
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, List, Optional

import requests

logger = logging.getLogger(__name__)

# ── Constants (self-contained -- do not depend on b3_pipeline/config.py) ────

DM_ENDPOINT = "https://seguro.bmfbovespa.com.br/rad/download/SolicitaDownload.asp"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DM_DATA_DIR = PROJECT_ROOT / "data" / "dm"

VALID_DOC_TYPES = {"TODOS", "RAD", "ITR", "DFP", "IAN", "IPE", "ENET"}

# CVM error code 22016 = "no records found" -- an empty result, not a failure.
ERROR_CODE_NO_RECORDS = "22016"

REQUEST_TIMEOUT = 60
DOWNLOAD_TIMEOUT = 300
RATE_LIMIT_DELAY = 0.3  # seconds between requests, be respectful to CVM's server

REGISTRATION_INSTRUCTIONS = """\
CVM_DM_USER / CVM_DM_PASS environment variables are not set.

The Download Múltiplo service requires free registration with CVM before it
issues credentials. To register:

  1. Go to CVM's external support portal: https://sistemas.cvm.gov.br/?suporte
  2. Submit a support request including:
       - Company or individual name
       - CNPJ or CPF
       - Complete address
       - Contact person's name
       - Contact person's job title
       - Contact person's phone number
       - Contact person's email address
       - Reason for requesting access to the download service
  3. CVM reviews the request and, once approved, emails a login and password
     to the contact email address provided.
  4. Set the credentials in your environment:
       export CVM_DM_USER="<login CVM sent you>"
       export CVM_DM_PASS="<password CVM sent you>"

Technical manual: https://conteudo.cvm.gov.br/menu/regulados/companhias/download_multiplo/manual_tecnico.html
See also docs/download_multiplo.md in this repo.
"""


class CredentialsMissingError(RuntimeError):
    """Raised when CVM_DM_USER / CVM_DM_PASS are not set in the environment."""

    def __init__(self):
        super().__init__(REGISTRATION_INSTRUCTIONS)


class DownloadMultiploError(RuntimeError):
    """Raised when CVM's Download Múltiplo endpoint returns an <ERROS> response."""


def _get_credentials() -> tuple:
    user = os.environ.get("CVM_DM_USER")
    password = os.environ.get("CVM_DM_PASS")
    if not user or not password:
        raise CredentialsMissingError()
    return user, password


def _date_range(start: date, end: date) -> Iterable[date]:
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


# ── Protocol calls ───────────────────────────────────────────────────────────

def request_links(single_date: date, doc_type: str) -> List[dict]:
    """Call SolicitaDownload.asp for one date + document type.

    Returns a list of dicts (one per <Link>: url, Documento, ccvm, DataRef,
    Situacao, ...). Returns [] when CVM reports "no records found" for that
    date/type. Raises CredentialsMissingError if env vars are unset, or
    DownloadMultiploError on any other CVM-reported error (e.g. bad login).
    """
    doc_type = doc_type.upper()
    if doc_type not in VALID_DOC_TYPES:
        raise ValueError(f"doc_type must be one of {sorted(VALID_DOC_TYPES)}, got {doc_type!r}")

    user, password = _get_credentials()
    payload = {
        "txtLogin": user,
        "txtSenha": password,
        "txtData": single_date.strftime("%d/%m/%Y"),
        "txtHora": "00:00",
        "txtDocumento": doc_type,
    }

    resp = requests.post(DM_ENDPOINT, data=payload, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()

    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError as e:
        raise DownloadMultiploError(f"Could not parse XML response for {single_date} {doc_type}: {e}") from e

    erro_num = root.findtext("NUMERO_DO_ERRO")
    if erro_num is not None:
        desc = root.findtext("DESCRICAO_DO_ERRO", default="unknown error")
        if erro_num == ERROR_CODE_NO_RECORDS:
            return []
        raise DownloadMultiploError(
            f"CVM Download Múltiplo error {erro_num} for {single_date} {doc_type}: {desc}"
        )

    return [dict(link_el.attrib) for link_el in root.findall(".//Link")]


def download_document(
    link: dict, force: bool = False, submission_date: Optional[date] = None
) -> Optional[Path]:
    """Download one <Link>'s file into data/dm/. Returns the saved Path, or None on failure.

    submission_date (the txtData search date this link came from) is baked into
    the filename: DataRef alone is the PERIOD reference date, so a restatement
    of the same period would collide with the original file and be skipped.
    """
    url = link.get("url")
    if not url:
        logger.warning(f"Link has no url attribute: {link}")
        return None

    doc = link.get("Documento", "DOC")
    ccvm = link.get("ccvm", "0")
    data_ref = (link.get("DataRef") or "").replace("/", "")
    sub = f"_{submission_date.strftime('%Y%m%d')}" if submission_date else ""
    suffix = Path(url.split("?")[0]).suffix or ".zip"
    filename = f"{doc}_{ccvm}_{data_ref}{sub}{suffix}"
    filepath = DM_DATA_DIR / filename

    if filepath.exists() and not force:
        logger.debug(f"File already exists: {filepath}")
        return filepath

    DM_DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        resp = requests.get(url, timeout=DOWNLOAD_TIMEOUT, stream=True)
        resp.raise_for_status()
        with open(filepath, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        logger.info(f"Downloaded {filepath}")
        return filepath
    except requests.RequestException as e:
        logger.error(f"Failed to download {url}: {e}")
        if filepath.exists():
            filepath.unlink()
        return None


def download_range(
    start_date: date,
    end_date: date,
    doc_types: Iterable[str] = ("ITR", "DFP", "IAN"),
    force: bool = False,
) -> List[Path]:
    """Request and download every filing package for doc_types over [start_date, end_date].

    Iterates one SolicitaDownload.asp call per (date, doc_type) pair -- the
    protocol has no date-range parameter, only a single txtData. Skips
    'Cancelado' links. Returns the list of successfully downloaded paths.
    """
    downloaded: List[Path] = []
    for single_date in _date_range(start_date, end_date):
        for doc_type in doc_types:
            try:
                links = request_links(single_date, doc_type)
            except DownloadMultiploError as e:
                logger.warning(str(e))
                continue
            for link in links:
                if link.get("Situacao", "Liberado") == "Cancelado":
                    continue
                path = download_document(link, force=force, submission_date=single_date)
                if path:
                    downloaded.append(path)
            time.sleep(RATE_LIMIT_DELAY)
    return downloaded
