# scrape.py
from __future__ import annotations
import re
from datetime import datetime
from pathlib import Path
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE = "https://web-scraping.dev"
OUT_DIR = Path("data")
OUT_DIR.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; HW3 bot; +https://example.com)"
}

def get_soup(url: str, headers: dict = None) -> BeautifulSoup:
    req_headers = HEADERS.copy()
    if headers:
        req_headers.update(headers)
    r = requests.get(url, headers=req_headers, timeout=30)
    r.raise_for_status()
    time.sleep(0.5)  # Be polite
    return BeautifulSoup(r.text, "lxml")

def parse_date(raw: str) -> datetime | None:
    raw = raw.strip()
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%b %d, %Y", "%B %d, %Y", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            pass
    return None

def scrape_products() -> pd.DataFrame:
    print("Scraping products from all categories...")
    all_rows = []
    seen_links = set()  # Track unique products to avoid duplicates

    # Scrape each category
    categories = ["apparel", "consumables", "household"]

    for category in categories:
        print(f"\n  Category: {category}")
        rows = []
        url = f"{BASE}/products?category={category}"
        page = 1

        while url:
            print(f"    Fetching page {page}...")
            soup = get_soup(url)

            # Find all product cards
            products = soup.select("div.row.product")
            print(f"    Found {len(products)} products on page {page}")

            for product in products:
                # Extract product name from h3 a
                name_el = product.select_one("h3 a")
                name = name_el.get_text(strip=True) if name_el else None

                # Extract product link
                link = name_el.get("href") if name_el else None
                if link and not link.startswith("http"):
                    link = BASE + link

                # Skip duplicates
                if link in seen_links:
                    continue
                seen_links.add(link)

                # Extract price from div.price
                price_el = product.select_one("div.price")
                price = price_el.get_text(strip=True) if price_el else None

                # Extract description
                desc_el = product.select_one("div.short-description")
                description = desc_el.get_text(strip=True) if desc_el else None

                rows.append({
                    "name": name,
                    "price": price,
                    "link": link,
                    "description": description,
                    "category": category,
                })

            # Find next page link (look for ">" link in pagination)
            next_link = None
            for a in soup.select("div.paging a"):
                if a.get_text(strip=True) == ">":
                    next_link = a.get("href")
                    break

            if next_link:
                url = next_link if next_link.startswith("http") else BASE + next_link
                page += 1
            else:
                url = None

        print(f"    Subtotal for {category}: {len(rows)} products")
        all_rows.extend(rows)

    print(f"\nTotal unique products scraped: {len(all_rows)}")
    return pd.DataFrame(all_rows)

def scrape_testimonials() -> pd.DataFrame:
    print("Scraping testimonials...")
    rows = []
    url = f"{BASE}/testimonials"
    page = 1

    # Secret token and referer required for HTMX API requests
    secret_headers = {
        "x-secret-token": "secret123",
        "referer": f"{BASE}/testimonials"
    }

    while url:
        print(f"  Fetching testimonials page {page}...")
        if page == 1:
            soup = get_soup(url)
        else:
            soup = get_soup(url, headers=secret_headers)

        # Find all testimonial cards
        testimonials = soup.select("div.testimonial")
        print(f"  Found {len(testimonials)} testimonials on page {page}")

        for testimonial in testimonials:
            # Extract author from identicon-svg username attribute
            author_el = testimonial.select_one("identicon-svg")
            author = author_el.get("username") if author_el else None

            # Extract testimonial text from p.text
            text_el = testimonial.select_one("p.text")
            text = text_el.get_text(strip=True) if text_el else None

            # Extract rating by counting stars (svg elements)
            rating_el = testimonial.select_one("span.rating")
            rating = len(rating_el.select("svg")) if rating_el else None

            # Only add if it has actual content (not the trigger element)
            if text:
                rows.append({
                    "author": author,
                    "text": text,
                    "rating": rating,
                })

        # Find next page from HTMX hx-get attribute on last testimonial
        # Look for div.testimonial with hx-get attribute
        next_link = None
        for testimonial in soup.select("div.testimonial[hx-get]"):
            hx_get = testimonial.get("hx-get")
            if hx_get:
                next_link = hx_get

        if next_link:
            url = next_link if next_link.startswith("http") else BASE + next_link
            page += 1
        else:
            url = None

    print(f"Total testimonials scraped: {len(rows)}")
    return pd.DataFrame(rows)

def scrape_reviews(products_df: pd.DataFrame = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    print("Scraping reviews from product pages...")
    all_rows = []

    # Get product URLs (either passed in or scrape them)
    if products_df is None:
        print("  Getting product URLs...")
        products_df = scrape_products()

    product_links = products_df["link"].tolist()

    print(f"  Found {len(product_links)} products to check for reviews")

    # Visit each product page and extract reviews
    for idx, product_url in enumerate(product_links, 1):
        print(f"  [{idx}/{len(product_links)}] Fetching reviews from {product_url}")

        soup = get_soup(product_url)

        # Find the reviews JSON data embedded in the page
        reviews_script = soup.find("script", {"id": "reviews-data", "type": "application/json"})

        if not reviews_script:
            print(f"    No reviews found on this page")
            continue

        # Parse the JSON
        try:
            import json
            reviews_json = json.loads(reviews_script.string)
        except Exception as e:
            print(f"    Error parsing reviews JSON: {e}")
            continue

        print(f"    Found {len(reviews_json)} reviews")

        # Extract ALL reviews (not filtering by year)
        for review in reviews_json:
            date_raw = review.get("date", "")
            date_parsed = parse_date(date_raw) if date_raw else None

            # Determine if review is from 2023
            is_from_2023 = date_parsed is not None and date_parsed.year == 2023

            all_rows.append({
                "review_id": review.get("id"),
                "product_url": product_url,
                "text": review.get("text"),
                "rating": review.get("rating"),
                "date_raw": date_raw,
                "date": date_parsed,
                "is_from_2023": is_from_2023,
            })

    print(f"Total reviews scraped: {len(all_rows)}")

    # Convert to DataFrame and ensure date column is datetime
    all_reviews_df = pd.DataFrame(all_rows)
    if not all_reviews_df.empty:
        all_reviews_df["date"] = pd.to_datetime(all_reviews_df["date"], errors="coerce")

    # Create filtered DataFrame with only 2023 reviews
    reviews_2023_df = all_reviews_df[all_reviews_df["is_from_2023"] == True].copy()
    print(f"2023 reviews: {len(reviews_2023_df)}")

    return all_reviews_df, reviews_2023_df

def main():
    print("Starting web scraping...")
    print("=" * 60)

    # Scrape products
    products = scrape_products()
    products_file = OUT_DIR / "products.csv"
    products.to_csv(products_file, index=False)
    print(f"✓ Saved {len(products)} products to {products_file}")
    print()

    # Scrape testimonials
    testimonials = scrape_testimonials()
    testimonials_file = OUT_DIR / "testimonials.csv"
    testimonials.to_csv(testimonials_file, index=False)
    print(f"✓ Saved {len(testimonials)} testimonials to {testimonials_file}")
    print()

    # Scrape reviews (pass products to avoid re-scraping)
    all_reviews, reviews_2023 = scrape_reviews(products_df=products)

    # Save all reviews with is_from_2023 flag
    all_reviews_file = OUT_DIR / "reviews_all.csv"
    all_reviews.to_csv(all_reviews_file, index=False)
    print(f"✓ Saved {len(all_reviews)} reviews (all years) to {all_reviews_file}")

    # Save 2023 reviews only
    reviews_file = OUT_DIR / "reviews.csv"
    reviews_2023.to_csv(reviews_file, index=False)
    print(f"✓ Saved {len(reviews_2023)} reviews (2023 only) to {reviews_file}")
    print()

    print("=" * 60)
    print("Scraping completed!")
    print(f"All files saved to: {OUT_DIR.absolute()}")
    print(f"  - {products_file.name}")
    print(f"  - {testimonials_file.name}")
    print(f"  - {all_reviews_file.name} (all reviews with is_from_2023 flag)")
    print(f"  - {reviews_file.name} (2023 reviews only)")

if __name__ == "__main__":
    main()
