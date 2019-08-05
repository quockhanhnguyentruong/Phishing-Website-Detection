import ipaddress
import re
import requests
from datetime import date
from dateutil.parser import parse as date_parse

# Calculates number of months
def diff_month(d1, d2):
    return (d1.year - d2.year) * 12 + d1.month - d2.month

# Generate data set by extracting the features from the URL
def generate_data_set(url):
	
    data_set = []

    # Converts the given URL into standard format
    if not re.match(r"^https?", url):
        url = "http://" + url
    
    # Stores the response of the given URL
    try:
        response = requests.get(url)
    except:
        response = ""

    # Extracts domain from the given URL
    domain = re.findall(r"://([^/]+)/?", url)[0]

    # Requests all the information about the domain
    whois_response = requests.get("https://www.whois.com/whois/"+domain)

    rank_checker_response = requests.post("https://www.checkpagerank.net/index.php", {
        "name": domain
    })

    # Extracts global rank of the website
    try:
        global_rank = int(re.findall(r"Global Rank: ([0-9]+)", rank_checker_response.text)[0])
    except:
        global_rank = -1

    # having_IP_Address (1) sai roi
    try:
        ipaddress.ip_address(url)
        data_set.append(1) #Phishing
    except:
        data_set.append(-1)
    
    # URL_Length (2)
    if len(url) < 54:
        data_set.append(-1)
    elif (len(url) >= 54 and len(url) <= 75):
        data_set.append(0)
    else:
        data_set.append(1) #Phishing
    
    # Shortining_Service (3)
    if re.findall("bit.ly", url):
        data_set.append(1) #Phishing
    else:
        data_set.append(-1)
	
	# having_At_Symbol (4)
    symbol=re.findall(r'@',url)
    if(len(symbol)==0):
        data_set.append(-1)
    else:
        data_set.append(1) #Phishing
	
	# double_slash_redirecting (5)
    symbol1=re.findall(r'/',url)
    if(len(symbol1)>7):
        data_set.append(1)
    else:
        data_set.append(-1)
    
    # Prefix_Suffix (6)
    if re.findall(r"//[^\-]+-[^\-]+", url):
        data_set.append(1)
	#if re.findall(r"http?://[^\-]+-[^\-]+/", url):
        #data_set.append(1)
    else:
        data_set.append(-1)
	

    # having_Sub_Domain (7)
    if len(re.findall("\.", url)) == 1:
        data_set.append(-1)
    elif len(re.findall("\.", url)) == 2:
        data_set.append(0)
    else:
        data_set.append(1)
    
    # SSLfinal_State
    #data_set.append(-1)

    # Domain_registeration_length
    #data_set.append(-1)

    # Favicon
    #data_set.append(-1)

    # port (8)
    try:
        port = domain.split(":")[1]
        if port:
            data_set.append(1)
        else:
            data_set.append(-1)
    except:
        data_set.append(-1)

    # HTTPS_token (9)
    if re.findall("https\-", domain):
        data_set.append(1)
    else:
        data_set.append(-1)

    # Request_URL
    #data_set.append(-1)

    # URL_of_Anchor
    #data_set.append(-1)

    # Links_in_tags
    #data_set.append(-1)

    # SFH
    #data_set.append(-1)

    # Submitting_to_email (10)
    if re.findall(r"[mail\(\)|mailto:?]", str(response)):
        data_set.append(1)
    else:
        data_set.append(-1)
	
    # Abnormal_URL (11)
    if response.text:
        data_set.append(1)
    else:
        data_set.append(-1)

    # Redirect (12)
    if len(response.history) <= 1:
        data_set.append(-1)
    elif (len(response.history) >= 2 and len(response.history) < 4):
        data_set.append(0)
    else:
        data_set.append(1)

    # on_mouseover (13)
    if re.findall("<script>.+onmouseover.+</script>", response.text):
        data_set.append(1)
    else:
        data_set.append(-1)

    # RightClick (14)
    if re.findall(r"event.button ?== ?2", response.text):
        data_set.append(1)
    else:
        data_set.append(-1)

    # popUpWidnow (15)
    if re.findall(r"alert\(", response.text):
        data_set.append(1)
    else:
        data_set.append(-1)

    # Iframe (16)
    if re.findall(r"[<iframe>|<frameBorder>]", response.text):
        data_set.append(1)
    else:
        data_set.append(-1)

    # age_of_domain (17)
    try:
        registration_date = re.findall(r'Registration Date:</div><div class="df-value">([^<]+)</div>', whois_response.text)[0]
        if diff_month(date.today(), date_parse(registration_date)) >= 6:
            data_set.append(-1)
        else:
            data_set.append(1)
    except:
        data_set.append(1)

    # DNSRecord
    #data_set.append(-1)

    # web_traffic (18) coi lai
    try:
        if global_rank > 0 and global_rank < 100000:
            data_set.append(-1)
        else:
            data_set.append(1)
    except:
        data_set.append(1)

    # Page_Rank (19) coi lai
    try:
        if global_rank > 0 and global_rank < 100000:
            data_set.append(-1)
        else:
            data_set.append(1)
    except:
        data_set.append(1)

    # Google_Index (20)
    try:
        if global_rank > 0 and global_rank < 100000:
            data_set.append(-1)
        else:
            data_set.append(1)
    except:
        data_set.append(1)

    # Links_pointing_to_page
    number_of_links = len(re.findall(r"<a href=", response.text))
    if number_of_links == 0:
        data_set.append(1)
    elif number_of_links <= 2:
        data_set.append(0)
    else:
        data_set.append(-1)

    # Statistical_report
    #data_set.append(-1)

    print (data_set)
	
    return data_set

