"""Improved training script with better sample data"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from src.phishing_detector.detector import PhishingDetector
from src.phishing_detector.features.extractor import FeatureExtractor
from config.settings import settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_improved_sample_data() -> pd.DataFrame:
    """Load improved sample training data with more diverse URLs"""
    logger.info("Generating improved sample training data...")

    # More legitimate URLs (popular, trusted sites)
    # IMPORTANT: Include both www and non-www versions, plus legitimate subdomains
    legitimate_urls = [
        # Search engines and portals (with and without www)
        "https://www.google.com",
        "https://google.com",
        "https://mail.google.com",
        "https://drive.google.com",
        "https://docs.google.com",
        "https://www.bing.com",
        "https://bing.com",
        "https://www.yahoo.com",
        "https://yahoo.com",
        "https://mail.yahoo.com",
        "https://duckduckgo.com",
        # Social media (with and without www)
        "https://www.facebook.com",
        "https://facebook.com",
        "https://www.twitter.com",
        "https://twitter.com",
        "https://www.linkedin.com",
        "https://linkedin.com",
        "https://www.instagram.com",
        "https://instagram.com",
        "https://www.reddit.com",
        "https://reddit.com",
        # Tech companies (with and without www, plus subdomains)
        "https://www.microsoft.com",
        "https://microsoft.com",
        "https://login.microsoft.com",
        "https://docs.microsoft.com",
        "https://azure.microsoft.com",
        "https://www.apple.com",
        "https://apple.com",
        "https://support.apple.com",
        "https://www.amazon.com",
        "https://amazon.com",
        "https://aws.amazon.com",
        "https://www.netflix.com",
        "https://netflix.com",
        "https://www.adobe.com",
        "https://adobe.com",
        # Development (with and without www)
        "https://www.github.com",
        "https://github.com",
        "https://gist.github.com",
        "https://stackoverflow.com",
        "https://www.python.org",
        "https://python.org",
        "https://docs.python.org",
        "https://www.tensorflow.org",
        "https://tensorflow.org",
        "https://www.docker.com",
        "https://docker.com",
        "https://hub.docker.com",
        # News and media
        "https://www.cnn.com",
        "https://cnn.com",
        "https://www.bbc.com",
        "https://bbc.com",
        "https://www.nytimes.com",
        "https://nytimes.com",
        "https://www.theguardian.com",
        "https://theguardian.com",
        # Education
        "https://www.wikipedia.org",
        "https://wikipedia.org",
        "https://en.wikipedia.org",
        "https://www.coursera.org",
        "https://coursera.org",
        "https://www.edx.org",
        "https://edx.org",
        "https://www.khanacademy.org",
        "https://khanacademy.org",
        # E-commerce
        "https://www.ebay.com",
        "https://ebay.com",
        "https://www.etsy.com",
        "https://etsy.com",
        "https://www.walmart.com",
        "https://walmart.com",
        "https://www.target.com",
        "https://target.com",
        # Finance
        "https://www.paypal.com",
        "https://paypal.com",
        "https://www.stripe.com",
        "https://stripe.com",
        "https://www.chase.com",
        "https://chase.com",
        "https://www.bankofamerica.com",
        "https://bankofamerica.com",
        # Cloud services
        "https://www.dropbox.com",
        "https://dropbox.com",
        "https://drive.google.com",
        "https://onedrive.live.com",
        "https://www.icloud.com",
        "https://icloud.com",
        # Others
        "https://www.spotify.com",
        "https://spotify.com",
        "https://www.zoom.us",
        "https://zoom.us",
        "https://www.slack.com",
        "https://slack.com",
        "https://www.trello.com",
        "https://trello.com",
        "https://www.notion.so",
        "https://notion.so",
        "https://www.medium.com",
        "https://medium.com",
        "https://www.wordpress.com",
        "https://wordpress.com",
        "https://www.blogger.com",
        "https://blogger.com",
        "https://www.tumblr.com",
        "https://tumblr.com",
        "https://www.pinterest.com",
        "https://pinterest.com",
        "https://www.twitch.tv",
        "https://twitch.tv",
        "https://www.youtube.com",
        "https://youtube.com",
        # Government sites
        "https://www.usa.gov",
        "https://usa.gov",
        "https://www.whitehouse.gov",
        "https://whitehouse.gov",
        "https://www.irs.gov",
        "https://irs.gov",
        "https://www.fbi.gov",
        "https://fbi.gov",
        "https://www.nasa.gov",
        "https://nasa.gov",
        "https://www.cdc.gov",
        "https://cdc.gov",
        "https://www.state.gov",
        "https://state.gov",
        # Educational institutions
        "https://www.mit.edu",
        "https://mit.edu",
        "https://www.stanford.edu",
        "https://stanford.edu",
        "https://www.harvard.edu",
        "https://harvard.edu",
        "https://www.berkeley.edu",
        "https://berkeley.edu",
        "https://www.yale.edu",
        "https://yale.edu",
        "https://www.princeton.edu",
        "https://princeton.edu",
        "https://www.columbia.edu",
        "https://columbia.edu",
        "https://www.cornell.edu",
        "https://cornell.edu",
        "https://www.upenn.edu",
        "https://upenn.edu",
        "https://www.caltech.edu",
        "https://caltech.edu",
        # More tech companies
        "https://www.oracle.com",
        "https://oracle.com",
        "https://www.salesforce.com",
        "https://salesforce.com",
        "https://www.ibm.com",
        "https://ibm.com",
        "https://www.intel.com",
        "https://intel.com",
        "https://www.amd.com",
        "https://amd.com",
        "https://www.nvidia.com",
        "https://nvidia.com",
        "https://www.vmware.com",
        "https://vmware.com",
        "https://www.cisco.com",
        "https://cisco.com",
        "https://www.dell.com",
        "https://dell.com",
        "https://www.hp.com",
        "https://hp.com",
        "https://www.lenovo.com",
        "https://lenovo.com",
        "https://www.samsung.com",
        "https://samsung.com",
        "https://www.sony.com",
        "https://sony.com",
        "https://www.lg.com",
        "https://lg.com",
        # More banks and financial
        "https://www.wellsfargo.com",
        "https://wellsfargo.com",
        "https://www.citi.com",
        "https://citi.com",
        "https://www.usbank.com",
        "https://usbank.com",
        "https://www.capitalone.com",
        "https://capitalone.com",
        "https://www.tdbank.com",
        "https://tdbank.com",
        "https://www.schwab.com",
        "https://schwab.com",
        "https://www.fidelity.com",
        "https://fidelity.com",
        "https://www.vanguard.com",
        "https://vanguard.com",
        "https://www.goldmansachs.com",
        "https://goldmansachs.com",
        "https://www.morganstanley.com",
        "https://morganstanley.com",
        # More e-commerce
        "https://www.bestbuy.com",
        "https://bestbuy.com",
        "https://www.homedepot.com",
        "https://homedepot.com",
        "https://www.lowes.com",
        "https://lowes.com",
        "https://www.costco.com",
        "https://costco.com",
        "https://www.samsclub.com",
        "https://samsclub.com",
        "https://www.aliexpress.com",
        "https://aliexpress.com",
        "https://www.alibaba.com",
        "https://alibaba.com",
        "https://www.wayfair.com",
        "https://wayfair.com",
        "https://www.overstock.com",
        "https://overstock.com",
        "https://www.zappos.com",
        "https://zappos.com",
        "https://www.newegg.com",
        "https://newegg.com",
        "https://www.macys.com",
        "https://macys.com",
        "https://www.nordstrom.com",
        "https://nordstrom.com",
        # More news and media
        "https://www.wsj.com",
        "https://wsj.com",
        "https://www.washingtonpost.com",
        "https://washingtonpost.com",
        "https://www.usatoday.com",
        "https://usatoday.com",
        "https://www.forbes.com",
        "https://forbes.com",
        "https://www.bloomberg.com",
        "https://bloomberg.com",
        "https://www.reuters.com",
        "https://reuters.com",
        "https://www.time.com",
        "https://time.com",
        "https://www.newsweek.com",
        "https://newsweek.com",
        "https://www.theverge.com",
        "https://theverge.com",
        "https://www.techcrunch.com",
        "https://techcrunch.com",
        "https://www.wired.com",
        "https://wired.com",
        "https://www.engadget.com",
        "https://engadget.com",
        "https://www.arstechnica.com",
        "https://arstechnica.com",
        "https://www.gizmodo.com",
        "https://gizmodo.com",
        # SaaS and productivity
        "https://www.atlassian.com",
        "https://atlassian.com",
        "https://www.asana.com",
        "https://asana.com",
        "https://www.monday.com",
        "https://monday.com",
        "https://www.airtable.com",
        "https://airtable.com",
        "https://www.figma.com",
        "https://figma.com",
        "https://www.canva.com",
        "https://canva.com",
        "https://www.miro.com",
        "https://miro.com",
        "https://www.lucidchart.com",
        "https://lucidchart.com",
        "https://www.hubspot.com",
        "https://hubspot.com",
        "https://www.mailchimp.com",
        "https://mailchimp.com",
        "https://www.zendesk.com",
        "https://zendesk.com",
        "https://www.intercom.com",
        "https://intercom.com",
        "https://www.freshworks.com",
        "https://freshworks.com",
        # Cloud and hosting
        "https://www.digitalocean.com",
        "https://digitalocean.com",
        "https://www.linode.com",
        "https://linode.com",
        "https://www.vultr.com",
        "https://vultr.com",
        "https://www.heroku.com",
        "https://heroku.com",
        "https://www.netlify.com",
        "https://netlify.com",
        "https://www.vercel.com",
        "https://vercel.com",
        "https://www.cloudflare.com",
        "https://cloudflare.com",
        "https://www.godaddy.com",
        "https://godaddy.com",
        "https://www.namecheap.com",
        "https://namecheap.com",
        "https://www.bluehost.com",
        "https://bluehost.com",
        "https://www.hostgator.com",
        "https://hostgator.com",
        # Communication and collaboration
        "https://www.teams.microsoft.com",
        "https://teams.microsoft.com",
        "https://www.webex.com",
        "https://webex.com",
        "https://www.gotomeeting.com",
        "https://gotomeeting.com",
        "https://www.discord.com",
        "https://discord.com",
        "https://www.telegram.org",
        "https://telegram.org",
        "https://www.signal.org",
        "https://signal.org",
        "https://www.whatsapp.com",
        "https://whatsapp.com",
        "https://www.skype.com",
        "https://skype.com",
        # Entertainment and streaming
        "https://www.hulu.com",
        "https://hulu.com",
        "https://www.disneyplus.com",
        "https://disneyplus.com",
        "https://www.hbomax.com",
        "https://hbomax.com",
        "https://www.primevideo.com",
        "https://primevideo.com",
        "https://www.peacocktv.com",
        "https://peacocktv.com",
        "https://www.paramountplus.com",
        "https://paramountplus.com",
        "https://www.crunchyroll.com",
        "https://crunchyroll.com",
        "https://www.soundcloud.com",
        "https://soundcloud.com",
        "https://www.pandora.com",
        "https://pandora.com",
        "https://www.deezer.com",
        "https://deezer.com",
        "https://www.tidal.com",
        "https://tidal.com",
        # Gaming
        "https://www.steam.com",
        "https://steam.com",
        "https://store.steampowered.com",
        "https://www.epicgames.com",
        "https://epicgames.com",
        "https://www.ea.com",
        "https://ea.com",
        "https://www.blizzard.com",
        "https://blizzard.com",
        "https://www.roblox.com",
        "https://roblox.com",
        "https://www.minecraft.net",
        "https://minecraft.net",
        "https://www.nintendo.com",
        "https://nintendo.com",
        "https://www.playstation.com",
        "https://playstation.com",
        "https://www.xbox.com",
        "https://xbox.com",
        # Travel and hospitality
        "https://www.airbnb.com",
        "https://airbnb.com",
        "https://www.booking.com",
        "https://booking.com",
        "https://www.expedia.com",
        "https://expedia.com",
        "https://www.hotels.com",
        "https://hotels.com",
        "https://www.tripadvisor.com",
        "https://tripadvisor.com",
        "https://www.marriott.com",
        "https://marriott.com",
        "https://www.hilton.com",
        "https://hilton.com",
        "https://www.hyatt.com",
        "https://hyatt.com",
        "https://www.united.com",
        "https://united.com",
        "https://www.delta.com",
        "https://delta.com",
        "https://www.southwest.com",
        "https://southwest.com",
        "https://www.aa.com",
        "https://aa.com",
        # Food delivery
        "https://www.ubereats.com",
        "https://ubereats.com",
        "https://www.doordash.com",
        "https://doordash.com",
        "https://www.grubhub.com",
        "https://grubhub.com",
        "https://www.postmates.com",
        "https://postmates.com",
        "https://www.instacart.com",
        "https://instacart.com",
        # Health and fitness
        "https://www.webmd.com",
        "https://webmd.com",
        "https://www.mayoclinic.org",
        "https://mayoclinic.org",
        "https://www.healthline.com",
        "https://healthline.com",
        "https://www.fitbit.com",
        "https://fitbit.com",
        "https://www.myfitnesspal.com",
        "https://myfitnesspal.com",
        "https://www.peloton.com",
        "https://peloton.com",
        # Job sites
        "https://www.linkedin.com/jobs",
        "https://www.indeed.com",
        "https://indeed.com",
        "https://www.glassdoor.com",
        "https://glassdoor.com",
        "https://www.monster.com",
        "https://monster.com",
        "https://www.ziprecruiter.com",
        "https://ziprecruiter.com",
        # More international sites
        "https://www.baidu.com",
        "https://baidu.com",
        "https://www.yandex.com",
        "https://yandex.com",
        "https://www.taobao.com",
        "https://taobao.com",
        "https://www.jd.com",
        "https://jd.com",
        "https://www.rakuten.com",
        "https://rakuten.com",
        "https://www.mercadolibre.com",
        "https://mercadolibre.com",
        # More development tools
        "https://www.gitlab.com",
        "https://gitlab.com",
        "https://www.bitbucket.org",
        "https://bitbucket.org",
        "https://www.npmjs.com",
        "https://npmjs.com",
        "https://www.pypi.org",
        "https://pypi.org",
        "https://www.packagist.org",
        "https://packagist.org",
        "https://www.maven.apache.org",
        "https://maven.apache.org",
        "https://www.jenkins.io",
        "https://jenkins.io",
        "https://www.travis-ci.com",
        "https://travis-ci.com",
        "https://www.circleci.com",
        "https://circleci.com",
        # Documentation sites
        "https://developer.mozilla.org",
        "https://www.w3schools.com",
        "https://w3schools.com",
        "https://www.geeksforgeeks.org",
        "https://geeksforgeeks.org",
        "https://www.javatpoint.com",
        "https://javatpoint.com",
        "https://www.tutorialspoint.com",
        "https://tutorialspoint.com",
        # More cloud providers
        "https://cloud.google.com",
        "https://www.oracle.com/cloud",
        "https://www.ibm.com/cloud",
        "https://www.alibabacloud.com",
        "https://alibabacloud.com",
        # Security and antivirus
        "https://www.norton.com",
        "https://norton.com",
        "https://www.mcafee.com",
        "https://mcafee.com",
        "https://www.kaspersky.com",
        "https://kaspersky.com",
        "https://www.avast.com",
        "https://avast.com",
        "https://www.avg.com",
        "https://avg.com",
        "https://www.bitdefender.com",
        "https://bitdefender.com",
        # Office and productivity
        "https://www.office.com",
        "https://office.com",
        "https://docs.google.com/spreadsheets",
        "https://docs.google.com/document",
        "https://docs.google.com/presentation",
        "https://www.zoho.com",
        "https://zoho.com",
        # More social media variants
        "https://www.tiktok.com",
        "https://tiktok.com",
        "https://www.snapchat.com",
        "https://snapchat.com",
        "https://www.vimeo.com",
        "https://vimeo.com",
        "https://www.dailymotion.com",
        "https://dailymotion.com",
        "https://www.flickr.com",
        "https://flickr.com",
        "https://www.deviantart.com",
        "https://deviantart.com",
        # More retailers
        "https://www.ikea.com",
        "https://ikea.com",
        "https://www.zara.com",
        "https://zara.com",
        "https://www.hm.com",
        "https://hm.com",
        "https://www.gap.com",
        "https://gap.com",
        "https://www.nike.com",
        "https://nike.com",
        "https://www.adidas.com",
        "https://adidas.com",
        "https://www.underarmour.com",
        "https://underarmour.com",
        "https://www.sephora.com",
        "https://sephora.com",
        "https://www.ulta.com",
        "https://ulta.com",
        # More education platforms
        "https://www.udemy.com",
        "https://udemy.com",
        "https://www.skillshare.com",
        "https://skillshare.com",
        "https://www.pluralsight.com",
        "https://pluralsight.com",
        "https://www.lynda.com",
        "https://lynda.com",
        "https://www.codecademy.com",
        "https://codecademy.com",
        "https://www.freecodecamp.org",
        "https://freecodecamp.org",
        # More subdomains of popular services
        "https://accounts.google.com",
        "https://calendar.google.com",
        "https://maps.google.com",
        "https://photos.google.com",
        "https://play.google.com",
        "https://news.google.com",
        "https://translate.google.com",
        "https://meet.google.com",
        "https://chat.google.com",
        "https://www.google.com/maps",
        "https://www.google.com/photos",
        "https://business.facebook.com",
        "https://developers.facebook.com",
        "https://m.facebook.com",
        "https://web.whatsapp.com",
        "https://api.whatsapp.com",
        "https://developer.apple.com",
        "https://store.apple.com",
        "https://music.apple.com",
        "https://tv.apple.com",
        "https://www.apple.com/store",
        "https://smile.amazon.com",
        "https://music.amazon.com",
        "https://video.amazon.com",
        "https://www.amazon.com/prime",
        "https://developer.amazon.com",
        "https://help.netflix.com",
        "https://www.netflix.com/browse",
        "https://partner.microsoft.com",
        "https://portal.azure.com",
        "https://outlook.office.com",
        "https://www.office.com/launch",
    ]

    # Phishing-like URLs (fake/suspicious patterns)
    phishing_urls = [
        # IP addresses
        "http://192.168.1.1/bank/login",
        "http://10.0.0.1/paypal/signin",
        "http://172.16.0.1/secure/account",
        # Suspicious TLDs
        "http://paypal-verify.suspicious-domain.tk/login.php",
        "http://apple-id-locked.ml/verify",
        "http://microsoft-security-alert.gq/signin",
        "http://banking-secure-login.click/auth",
        "http://amazon.security-check.top/account",
        "http://ebay-suspended.racing/verify",
        "http://google-security.download/signin",
        "http://facebook-verify.link/confirm",
        "http://netflix-payment.bid/update",
        "http://linkedin-security.date/verify",
        "http://twitter-suspended.accountant/appeal",
        "http://instagram-verify.science/confirm",
        "http://dropbox-storage-full.win/upgrade",
        "http://adobe-update-required.xyz/download",
        "http://spotify-premium-free.top/claim",
        "http://apple-icloud-storage.work/upgrade",
        # Misspellings and typos
        "http://gooogle.com/login",
        "http://faceboook.com/signin",
        "http://amaz0n.com/account",
        "http://paypa1.com/verify",
        "http://app1e.com/icloud",
        # Suspicious keywords
        "http://secure-paypal-update.com/login",
        "http://account-verify-amazon.net/confirm",
        "http://microsoft-security-team.org/alert",
        "http://apple-account-locked.info/unlock",
        "http://netflix-billing-update.biz/payment",
        "http://bank-account-suspended.online/reactivate",
        # URL shorteners with suspicious patterns
        "http://bit.ly/3xYz@suspicious",
        "http://tinyurl.com/verify-account-now",
        # Subdomain spoofing
        "http://paypal.com-verify.work/update",
        "http://apple.com-support.xyz/help",
        "http://amazon.com-secure.top/account",
        "http://microsoft.com-login.info/signin",
        # Long suspicious URLs
        "http://verify-your-paypal-account-now-or-suspended.tk/login.php",
        "http://amazon-account-verification-required-urgent.ml/verify",
        "http://apple-id-security-alert-action-needed.gq/confirm",
        # Double slashes and special chars
        "http://paypal.com//login",
        "http://amazon.com@phishing.tk",
        "http://secure-login.com//..//bank",
        # Phishing with ports
        "http://paypal.com:8080/login",
        "http://amazon.com:3000/account",
        # Mixed legitimate and suspicious
        "http://secure-update-microsoft.tk/signin",
        "http://verify-google-account.ml/confirm",
        "http://apple-support-team.gq/help",
        "http://netflix-reactivate.xyz/payment",
    ]

    # Create DataFrame
    df = pd.DataFrame(
        {
            "url": legitimate_urls + phishing_urls,
            "label": [0] * len(legitimate_urls) + [1] * len(phishing_urls),
        }
    )

    logger.info(f"Generated {len(legitimate_urls)} legitimate URLs")
    logger.info(f"Generated {len(phishing_urls)} phishing URLs")
    logger.info(f"Total: {len(df)} URLs")

    return df


def extract_features_from_urls(urls: list, labels: list) -> tuple:
    """Extract features from URLs"""
    logger.info("Extracting features from URLs...")

    feature_extractor = FeatureExtractor(
        enable_dns_lookup=False,  # Disable for faster training
        enable_whois_lookup=False,
        timeout=5,
    )

    # Extract features
    features_list = []
    valid_labels = []

    for url, label in zip(urls, labels):
        try:
            features = feature_extractor.extract_features(url)
            features_list.append(features)
            # Ensure label is integer
            valid_labels.append(int(label))
        except Exception as e:
            logger.warning(f"Failed to extract features from {url}: {e}")

    # Convert to DataFrame
    df_features = pd.DataFrame(features_list)

    # Get feature names
    feature_names = df_features.columns.tolist()

    # Convert to numpy arrays
    X = df_features.values
    y = np.array(valid_labels, dtype=np.int32)

    logger.info(f"Extracted {X.shape[1]} features from {X.shape[0]} URLs")

    return X, y, feature_names


def evaluate_model(detector: PhishingDetector, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate model performance"""
    logger.info("Evaluating model...")

    # Get predictions
    y_pred_proba = detector.ensemble_classifier.predict_proba(X_test)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
    }

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    logger.info("\n" + "=" * 50)
    logger.info("MODEL EVALUATION RESULTS")
    logger.info("=" * 50)
    logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall:    {metrics['recall']:.4f}")
    logger.info(f"F1-Score:  {metrics['f1_score']:.4f}")
    logger.info(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    logger.info("\nConfusion Matrix:")
    logger.info(f"TN: {cm[0][0]}, FP: {cm[0][1]}")
    logger.info(f"FN: {cm[1][0]}, TP: {cm[1][1]}")
    logger.info("=" * 50)

    # Classification report
    logger.info("\nClassification Report:")
    logger.info("\n" + classification_report(y_test, y_pred, target_names=["Legitimate", "Phishing"]))

    return metrics


def train_model(
    data_path: str = None,
    output_dir: str = None,
    test_size: float = 0.2,
    random_state: int = 42,
    max_rows: int = None,
) -> None:
    """Main training function"""

    # Use settings defaults if not provided
    output_dir = output_dir or settings.model_save_path
    random_state = random_state or settings.random_seed

    # Load data
    if data_path and os.path.exists(data_path):
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path, on_bad_lines='skip')

        # Limit rows if specified
        if max_rows:
            logger.info(f"Limiting to first {max_rows} rows")
            df = df.head(max_rows)

        # Convert labels to integers
        df['label'] = pd.to_numeric(df['label'], errors='coerce')
        df = df.dropna(subset=['label'])
        df['label'] = df['label'].astype(int)

        # Filter valid labels (0 or 1)
        df = df[df['label'].isin([0, 1])]

        logger.info(f"Loaded {len(df)} total URLs")
        logger.info(f"  Legitimate (0): {len(df[df['label'] == 0])}")
        logger.info(f"  Phishing (1): {len(df[df['label'] == 1])}")

        urls = df["url"].tolist()
        labels = df["label"].tolist()
    else:
        logger.info("Using improved sample data")
        df = load_improved_sample_data()
        urls = df["url"].tolist()
        labels = df["label"].tolist()

    # Extract features
    X, y, feature_names = extract_features_from_urls(urls, labels)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info(f"Training set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    logger.info(f"Phishing ratio in training: {y_train.mean():.2%}")

    # Initialize detector with adjusted settings
    detector = PhishingDetector(
        enable_zero_day=True,
        enable_dns_lookup=False,
        anomaly_threshold=0.5,  # Increased threshold to reduce false positives
        random_state=random_state,
    )

    # Train model
    logger.info("Starting training...")
    history = detector.train(X_train, y_train, X_test, y_test, feature_names)

    # Evaluate model
    metrics = evaluate_model(detector, X_test, y_test)

    # Get feature importance
    logger.info("\nTop 10 Most Important Features:")
    feature_importance = detector.get_feature_importance()
    for i, (feature, importance) in enumerate(list(feature_importance.items())[:10], 1):
        logger.info(f"{i}. {feature}: {importance:.4f}")

    # Save model
    logger.info(f"\nSaving model to {output_dir}")
    detector.save(output_dir)

    logger.info("\nTraining completed successfully!")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Train phishing detector model with improved data")
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to training data CSV file (url, label columns)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for trained models",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set size (default: 0.2)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Maximum number of rows to use from CSV (default: all)",
    )

    args = parser.parse_args()

    try:
        train_model(
            data_path=args.data,
            output_dir=args.output,
            test_size=args.test_size,
            random_state=args.random_state,
            max_rows=args.max_rows,
        )
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
