import base64
import json
import requests


def test_b3(payload_dict, endpoint):
    url = f"https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/{endpoint}/{{}}"
    json_str = json.dumps(payload_dict, separators=(",", ":"))
    encoded = base64.b64encode(json_str.encode("utf-8")).decode("utf-8")
    resp = requests.get(
        url.format(encoded),
        headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
    )
    print(f"{endpoint} [{resp.status_code}]:")
    try:
        print(str(resp.json())[:200])
    except:
        print(resp.text[:200])


print("Test 1: Cash Dividends with issuingCompany")
test_b3(
    {"issuingCompany": "PETR", "language": "pt-br", "pageNumber": 1, "pageSize": 10},
    "GetListedCashDividends",
)

print("\nTest 2: Cash Dividends with tradingName")
test_b3(
    {"tradingName": "PETROBRAS", "language": "pt-br", "pageNumber": 1, "pageSize": 10},
    "GetListedCashDividends",
)

print("\nTest 3: Supplement with issuingCompany PETR")
test_b3({"issuingCompany": "PETR", "language": "pt-br"}, "GetListedSupplementCompany")

print("\nTest 4: Supplement with tradingName PETROBRAS")
test_b3({"tradingName": "PETROBRAS", "language": "pt-br"}, "GetListedSupplementCompany")
