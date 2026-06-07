import os
import csv, json, time, hashlib, asyncio, aiohttp, aiofiles, re, requests
from pathlib import Path
from tqdm import tqdm

SECRET_KEY = os.environ.get("PEXELS_SECRET_KEY", "")
OUTPUT_DIR  = "pexels_dataset"
PER_KEYWORD = 500
PER_PAGE    = 24
CONCURRENT  = 8
API_DELAY   = 0.5
BASE_URL    = "https://www.pexels.com/en-us/api/v3/search/photos"

HEADERS = {
    "secret-key":      SECRET_KEY,
    "x-client-type":   "react",
    "referer":         "https://www.pexels.com/en-us/search/cat/",
    "user-agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36",
    "accept":          "*/*",
    "accept-language": "en-US,en;q=0.9",
    "dnt":             "1",
}

KEYWORDS = [
    "cat animal", "persian cat", "siamese cat", "bengal cat portrait",
    "tabby cat", "kitten cute", "black cat", "white cat",
    "dog animal", "golden retriever", "german shepherd", "husky dog",
    "labrador dog", "poodle dog", "bulldog pet", "beagle dog",
    "dalmatian dog", "border collie", "rottweiler dog", "chihuahua",
    "lion wildlife", "tiger wildlife", "cheetah running", "leopard wildlife",
    "jaguar wildlife", "snow leopard", "cougar mountain lion", "lynx wildlife",
    "bear wildlife", "grizzly bear fishing", "polar bear arctic",
    "black bear forest", "panda bear", "koala bear",
    "bird wildlife", "bald eagle flying", "owl bird", "penguin antarctica",
    "flamingo bird", "parrot colorful", "toucan bird", "peacock bird",
    "hummingbird flower", "hawk bird", "pelican bird", "crane bird",
    "swan lake", "duck water", "woodpecker bird", "duck pond", "mallard duck",
    "fish underwater", "dolphin ocean", "shark underwater", "whale ocean",
    "sea turtle", "octopus underwater", "jellyfish ocean", "seal animal",
    "walrus arctic", "otter animal", "clownfish coral",
    "monkey animal", "gorilla wildlife", "chimpanzee", "orangutan wildlife",
    "baboon africa", "gibbon primate",
    "elephant wildlife", "giraffe savanna", "zebra africa", "hippo wildlife",
    "rhinoceros wildlife", "wildebeest migration", "meerkat animal",
    "hyena wildlife", "warthog africa",
    "deer wildlife", "elk wildlife", "moose animal", "reindeer snow",
    "antelope africa", "bison wildlife", "ram sheep mountain",
    "crocodile reptile", "lizard reptile", "chameleon colorful",
    "snake reptile", "iguana lizard", "frog amphibian", "tortoise animal",
    "komodo dragon",
    "rabbit animal", "fox wildlife", "wolf forest", "squirrel animal",
    "raccoon animal", "hedgehog cute", "ferret animal", "hamster pet",
    "beaver wildlife", "porcupine animal",
    "horse animal", "wild horse mustang", "donkey animal", "goat farm",
    "sheep farm", "pig farm", "cow farm", "rooster chicken",
    "butterfly flower", "dragonfly insect", "bee flower",
    "ladybug insect", "mantis insect",
]

seen_hashes = set()
seen_ids    = set()
seen_lock   = asyncio.Lock()


def build_caption(attr, keyword):
    desc    = (attr.get("description") or "").strip()
    alt     = (attr.get("alt") or "").strip()
    caption = desc if desc and len(desc) > 10 else alt
    if not caption:
        caption = f"A photo of {keyword}"
    return re.sub(r'\s+', ' ', caption).strip()[:500]

def is_valid_caption(text):
    if not text or len(text.split()) < 8:
        return False
    if len(text) > 500 or '©' in text or 'copyright' in text.lower():
        return False
    return True

def pexels_search(keyword, page=1):
    params = {"query": keyword, "page": page, "per_page": PER_PAGE, "seo_tags": "true"}
    hdrs   = {**HEADERS, "referer": f"https://www.pexels.com/en-us/search/{keyword.replace(' ', '-')}/"}
    r      = requests.get(BASE_URL, params=params, headers=hdrs, timeout=20)

    if r.status_code == 429:
        print(f"\n  rate limited, waiting 60s...")
        time.sleep(60)
        return [], 0
    if r.status_code != 200:
        print(f"\n  status {r.status_code} on '{keyword}' page {page}")
        return [], 0
    try:
        data = r.json()
    except:
        return [], 0

    return data.get("data", []), data.get("pagination", {}).get("total_pages", 1)


async def download_one(session, task, semaphore, pbar):
    url, filepath, meta = task
    async with semaphore:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=20)) as r:
                if r.status != 200:
                    return None
                content = await r.read()
                if len(content) < 1000:
                    return None

                img_hash = hashlib.md5(content).hexdigest()
                async with seen_lock:
                    if img_hash in seen_hashes:
                        return None
                    seen_hashes.add(img_hash)

                async with aiofiles.open(filepath, "wb") as f:
                    await f.write(content)

                meta["hash"] = img_hash
                pbar.update(1)
                return meta
        except:
            return None


def collect_tasks(keyword, total):
    safe_kw = keyword.replace(" ", "_")
    img_dir = Path(OUTPUT_DIR) / safe_kw
    img_dir.mkdir(parents=True, exist_ok=True)

    tasks, page = [], 1
    while len(tasks) < total:
        try:
            photos, total_pages = pexels_search(keyword, page)
        except Exception as e:
            print(f"\n  api error: {e}")
            break

        if not photos:
            break

        for photo in photos:
            if len(tasks) >= total:
                break
            attr     = photo.get("attributes", {})
            photo_id = str(photo.get("id", ""))
            if photo_id in seen_ids:
                continue
            seen_ids.add(photo_id)

            img_url = attr.get("image", {}).get("medium")
            if not img_url:
                continue

            caption = build_caption(attr, keyword)
            if not is_valid_caption(caption):
                continue

            idx      = len(tasks) + 1
            filename = f"{safe_kw}_{idx:05d}.jpg"
            tasks.append((img_url, str(img_dir / filename), {
                "file":    str(Path(safe_kw) / filename),
                "title":   (attr.get("title") or "").strip(),
                "caption": caption,
                "keyword": keyword,
                "source":  "pexels",
                "id":      photo_id,
                "hash":    "",
            }))

        if page >= total_pages:
            break
        page += 1
        time.sleep(API_DELAY)

    return tasks


async def download_batch(tasks, pbar):
    semaphore = asyncio.Semaphore(CONCURRENT)
    connector = aiohttp.TCPConnector(limit=CONCURRENT, ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        results = await asyncio.gather(*[download_one(session, t, semaphore, pbar) for t in tasks])
    return [r for r in results if r]


def save_checkpoint(all_meta):
    csv_path = Path(OUTPUT_DIR) / "metadata.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "title", "caption", "keyword", "source", "id", "hash"])
        w.writeheader()
        w.writerows(all_meta)
    with open(Path(OUTPUT_DIR) / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(all_meta, f, ensure_ascii=False, indent=2)


async def main():
    import pandas as pd
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    all_meta, done_keywords = [], set()
    csv_path = Path(OUTPUT_DIR) / "metadata.csv"

    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        all_meta = existing.to_dict("records")
        done_keywords = set(existing["keyword"].unique())
        seen_ids.update(existing["id"].astype(str).tolist())
        print(f"resume: {len(done_keywords)} keywords done, {len(all_meta)} photos")
    else:
        print("starting fresh")

    remaining = [kw for kw in KEYWORDS if kw not in done_keywords]
    print(f"remaining: {len(remaining)} keywords\n")

    with tqdm(total=len(KEYWORDS) * PER_KEYWORD, initial=len(all_meta), unit="img") as pbar:
        for i, keyword in enumerate(remaining):
            pbar.set_description(f"[{len(done_keywords)+i+1}/{len(KEYWORDS)}] {keyword}")
            try:
                tasks   = collect_tasks(keyword, PER_KEYWORD)
                results = await download_batch(tasks, pbar)
                all_meta.extend(results)
                done_keywords.add(keyword)
                save_checkpoint(all_meta)
            except Exception as e:
                print(f"\n  error on '{keyword}': {e}, saving checkpoint...")
                save_checkpoint(all_meta)
                break
            time.sleep(1)

    print(f"\ndone: {len(all_meta)} photos, {len(seen_hashes)} unique hashes")

if __name__ == "__main__":
    asyncio.run(main())