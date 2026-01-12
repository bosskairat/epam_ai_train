from playwright.sync_api import sync_playwright

def get_top3_resumes(url: str):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="networkidle")
        
        # подождем 15 секунд максимум
        try:
            page.wait_for_selector('.resume-search-item', timeout=15000)
        except:
            browser.close()
            return {"error": "No resumes found or page blocked"}
        
        items = page.query_selector_all('.resume-search-item')[:3]
        resumes = []
        for item in items:
            title_el = item.query_selector('a.bloko-link')
            title = title_el.inner_text().strip() if title_el else ''
            link = title_el.get_attribute('href') if title_el else ''
            resumes.append({'title': title, 'link': link})
        
        browser.close()
        return resumes
