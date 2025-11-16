"""
Whitelist of known legitimate domains (Second-level domain + TLD only)
URLs from these domains will be automatically classified as SAFE regardless of path
"""

# Extracted from training data - only SLD + TLD
LEGITIMATE_DOMAINS = {
    # Search engines
    "google.com",
    "bing.com",
    "yahoo.com",
    "duckduckgo.com",

    # Social media
    "facebook.com",
    "twitter.com",
    "linkedin.com",
    "instagram.com",
    "reddit.com",

    # Tech companies
    "microsoft.com",
    "apple.com",
    "amazon.com",
    "netflix.com",
    "adobe.com",

    # Development
    "github.com",
    "stackoverflow.com",
    "python.org",
    "tensorflow.org",
    "docker.com",

    # News
    "cnn.com",
    "bbc.com",
    "nytimes.com",
    "theguardian.com",

    # Education
    "wikipedia.org",
    "coursera.org",
    "edx.org",
    "khanacademy.org",

    # E-commerce
    "ebay.com",
    "etsy.com",
    "walmart.com",
    "target.com",

    # Finance
    "paypal.com",
    "stripe.com",
    "chase.com",
    "bankofamerica.com",
    "wellsfargo.com",
    "citi.com",
    "usbank.com",
    "capitalone.com",
    "tdbank.com",
    "schwab.com",
    "fidelity.com",
    "vanguard.com",
    "goldmansachs.com",
    "morganstanley.com",

    # Cloud services
    "dropbox.com",
    "icloud.com",

    # Others
    "spotify.com",
    "zoom.us",
    "slack.com",
    "trello.com",
    "notion.so",
    "medium.com",
    "wordpress.com",
    "blogger.com",
    "tumblr.com",
    "pinterest.com",
    "twitch.tv",
    "youtube.com",

    # Government
    "usa.gov",
    "whitehouse.gov",
    "irs.gov",
    "fbi.gov",
    "nasa.gov",
    "cdc.gov",
    "state.gov",

    # Universities
    "mit.edu",
    "stanford.edu",
    "harvard.edu",
    "berkeley.edu",
    "yale.edu",
    "princeton.edu",
    "columbia.edu",
    "cornell.edu",
    "upenn.edu",
    "caltech.edu",

    # More tech
    "oracle.com",
    "salesforce.com",
    "ibm.com",
    "intel.com",
    "amd.com",
    "nvidia.com",
    "vmware.com",
    "cisco.com",
    "dell.com",
    "hp.com",
    "lenovo.com",
    "samsung.com",
    "sony.com",
    "lg.com",

    # More e-commerce
    "bestbuy.com",
    "homedepot.com",
    "lowes.com",
    "costco.com",
    "samsclub.com",
    "aliexpress.com",
    "alibaba.com",
    "wayfair.com",
    "overstock.com",
    "zappos.com",
    "newegg.com",
    "macys.com",
    "nordstrom.com",

    # More news
    "wsj.com",
    "washingtonpost.com",
    "usatoday.com",
    "forbes.com",
    "bloomberg.com",
    "reuters.com",
    "time.com",
    "newsweek.com",
    "theverge.com",
    "techcrunch.com",
    "wired.com",
    "engadget.com",
    "arstechnica.com",
    "gizmodo.com",

    # SaaS
    "atlassian.com",
    "asana.com",
    "monday.com",
    "airtable.com",
    "figma.com",
    "canva.com",
    "miro.com",
    "lucidchart.com",
    "hubspot.com",
    "mailchimp.com",
    "zendesk.com",
    "intercom.com",
    "freshworks.com",

    # Cloud/hosting
    "digitalocean.com",
    "linode.com",
    "vultr.com",
    "heroku.com",
    "netlify.com",
    "vercel.com",
    "cloudflare.com",
    "godaddy.com",
    "namecheap.com",
    "bluehost.com",
    "hostgator.com",

    # Communication
    "webex.com",
    "gotomeeting.com",
    "discord.com",
    "telegram.org",
    "signal.org",
    "whatsapp.com",
    "skype.com",

    # Streaming
    "hulu.com",
    "disneyplus.com",
    "hbomax.com",
    "primevideo.com",
    "peacocktv.com",
    "paramountplus.com",
    "crunchyroll.com",
    "soundcloud.com",
    "pandora.com",
    "deezer.com",
    "tidal.com",

    # Gaming
    "steam.com",
    "steampowered.com",
    "epicgames.com",
    "ea.com",
    "blizzard.com",
    "roblox.com",
    "minecraft.net",
    "nintendo.com",
    "playstation.com",
    "xbox.com",

    # Travel
    "airbnb.com",
    "booking.com",
    "expedia.com",
    "hotels.com",
    "tripadvisor.com",
    "marriott.com",
    "hilton.com",
    "hyatt.com",
    "united.com",
    "delta.com",
    "southwest.com",
    "aa.com",

    # Food delivery
    "ubereats.com",
    "doordash.com",
    "grubhub.com",
    "postmates.com",
    "instacart.com",

    # Health
    "webmd.com",
    "mayoclinic.org",
    "healthline.com",
    "fitbit.com",
    "myfitnesspal.com",
    "peloton.com",

    # Job sites
    "indeed.com",
    "glassdoor.com",
    "monster.com",
    "ziprecruiter.com",

    # International
    "baidu.com",
    "yandex.com",
    "taobao.com",
    "jd.com",
    "rakuten.com",
    "mercadolibre.com",

    # Dev tools
    "gitlab.com",
    "bitbucket.org",
    "npmjs.com",
    "pypi.org",
    "packagist.org",
    "jenkins.io",
    "circleci.com",

    # Docs
    "w3schools.com",
    "geeksforgeeks.org",
    "javatpoint.com",
    "tutorialspoint.com",

    # Security
    "norton.com",
    "mcafee.com",
    "kaspersky.com",
    "avast.com",
    "avg.com",
    "bitdefender.com",

    # Office
    "office.com",
    "zoho.com",

    # More social
    "tiktok.com",
    "snapchat.com",
    "vimeo.com",
    "dailymotion.com",
    "flickr.com",
    "deviantart.com",

    # Retail
    "ikea.com",
    "zara.com",
    "hm.com",
    "gap.com",
    "nike.com",
    "adidas.com",
    "underarmour.com",
    "sephora.com",
    "ulta.com",

    # Education platforms
    "udemy.com",
    "skillshare.com",
    "pluralsight.com",
    "lynda.com",
    "codecademy.com",
    "freecodecamp.org",
}
