import base64, json, requests

def fetch_company_data(trading_name: str):
    payload = {"issuingCompany": trading_name, "language": "pt-br"}
    json_str = json.dumps(payload, separators=(",", ":"))
    encoded = base64.b64encode(json_str.encode("utf-8")).decode("utf-8")
    url = f"https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedSupplementCompany/{encoded}"
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    return resp.json()

print(json.dumps(fetch_company_data("ITUB"), indent=2))
