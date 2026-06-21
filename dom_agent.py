import time
from playwright.sync_api import sync_playwright

def run_dom_agents():
    print("Starting DOM Agents for missing data...")
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            
            # 1. RBI DBIE
            print("Accessing RBI DBIE...")
            page = browser.new_page()
            try:
                page.goto("https://dbie.rbi.org.in/DBIE/dbie.rbi?site=statistics", timeout=30000)
                time.sleep(5)
                with open("data/raw/dbie_page.html", "w") as f:
                    f.write(page.content())
                print("Captured RBI DBIE DOM.")
            except Exception as e:
                print(f"RBI DBIE Error: {e}")
            page.close()

            # 2. MoSPI PLFS
            print("Accessing MoSPI PLFS...")
            page = browser.new_page()
            try:
                page.goto("https://www.mospi.gov.in/web/mospi/reports-publications", timeout=30000)
                time.sleep(5)
                links = page.query_selector_all("a")
                for l in links:
                    text = l.inner_text().strip().lower()
                    if "plfs" in text or "labour force" in text:
                        print(f"Found PLFS Link: {text} -> {l.get_attribute('href')}")
            except Exception as e:
                print(f"MoSPI Error: {e}")
            page.close()

            # 3. Tradestat
            print("Accessing Tradestat...")
            page = browser.new_page()
            try:
                page.goto("https://tradestat.commerce.gov.in", timeout=30000)
                time.sleep(5)
                with open("data/raw/tradestat_page.html", "w") as f:
                    f.write(page.content())
                print("Captured Tradestat DOM.")
            except Exception as e:
                print(f"Tradestat Error: {e}")
            page.close()

            browser.close()
    except Exception as e:
        print(f"Playwright Initialization Error: {e}")

if __name__ == "__main__":
    run_dom_agents()
