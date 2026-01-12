from fastapi import FastAPI, Query
from app.playwright_scraper import get_top3_resumes

app = FastAPI(title="HH.kz Resume Scraper API")



def build_hh_resume_search_url(search_query, experience, schedule, location):
    """
    Build hh.kz resume search URL without any imports.
    """

    # City → subdomain + area_id mapping
    area_map = {
        "Astana": {"subdomain": "astana.", "id": 159},
        "Almaty": {"subdomain": "almaty.", "id": 160},
        "Shymkent": {"subdomain": "shymkent.", "id": 221},
    }

    # fallback
    area = area_map.get(location, {"subdomain": "", "id": None})

    # simple URL encoding for spaces and basic symbols
    def simple_encode(s):
        return s.replace(" ", "+").replace("#", "%23").replace("&", "%26")

    encoded_query = simple_encode(search_query)

    # base parameters
    params = [
        f"text={encoded_query}",
        f"schedule={schedule}",
        f"experience={experience}",
        "currency_code=KZT",
        "ored_clusters=true",
        "order_by=relevance",
        "search_period=0",
        "logic=normal",
        "pos=full_text",
        "exp_period=all_time"
    ]

    # add area if exists
    if area["id"] is not None:
        params.append(f"area={area['id']}")
        params.append("isDefaultArea=true")

    # join all params
    query_string = "&".join(params)

    url = f"https://{area['subdomain']}hh.kz/search/resume?{query_string}"
    return url


@app.get("/scrape")
def scrape_hh(
    search_query: str = Query(..., description="Search query"),
    location: int = Query(159, description="Area ID (Astana=159)"),
    experience: str = Query("between3And6", description="Experience filter"),
    schedule: str = Query("remote", description="Work schedule")
):
    # Build HH.kz URL
    url = build_hh_resume_search_url(search_query, experience, schedule, location)    
    top3 = get_top3_resumes(url)

    return {"top3_resumes": top3}
