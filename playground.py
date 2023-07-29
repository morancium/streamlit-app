import requests
from bs4 import BeautifulSoup
import json
url = "https://docs.vitaracharts.com/guideMapFeatures/defaultMaps.html"
landing="https://docs.vitaracharts.com/guideAllCharts/about.html"
base="https://docs.vitaracharts.com"
tab_links=[]
response = requests.get(landing)
soup = BeautifulSoup(response.content, "lxml")

# tabs= soup.find_all("nav",class_="nav nav-tabs")
# tab_links_outer=tabs[0].find_all("a",class_="nav-item nav-link",href=True)
# for i in tab_links_outer:
#     tab_links.append(base+i['href'])
# print(tab_links)
tab_links=["https://docs.vitaracharts.com/guideAllCharts/about.html","https://docs.vitaracharts.com/guideGridFeatures/appearance.html","https://docs.vitaracharts.com/guideIBCSCommonFeatures/about.html","https://docs.vitaracharts.com/guideMapFeatures/defaultMaps.html","https://docs.vitaracharts.com/customization/about.html"]
Meta_texts=[]
Meta_json=[]
for url in tab_links:
    response = requests.get(url)
    response.raise_for_status()  # Check if the request was successful
    urls=[]
    title=[]
    soup = BeautifulSoup(response.content, "lxml")
    tags= soup.find_all("nav",class_="nav nav-pills leftMenu")
    for i in range(len(tags)):
        m=tags[i].find_all("a",href=True)
        for j in range(len(m)):
            title.append(m[j].text)
            print((m[j].text))
            urls.append(base+m[j]['href'])
    print(len(urls))
    print(len(title))
    for i, url in enumerate(urls):
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "lxml")
        texts=soup.find_all("div",class_="col-md-9 rightContent")
        body=texts[0].text
        body=body.replace('\n','')
        body=body.replace('\u2018','\'')
        body=body.replace('\u2019','\'')
        body=body.replace('\t','')
        Meta_json.append({
            "title":title[i],
            "body":body,
            "url":url
        })
        Meta_texts.append(body)
        # break
    json_data = json.dumps(Meta_json, indent=4)
    with open('output4.json', 'w') as outfile:
        json.dump(Meta_json, outfile)
    # print(Meta_json)
