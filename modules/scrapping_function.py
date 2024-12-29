import time
import pandas as pd
import numpy as np
import random

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager


def add_expansion(expansion_name):
    existent_data = pd.read_csv('data/original_data.csv')
    if expansion_name in existent_data["Expansion"].unique().tolist():
        pass
    else:
        titles = ["Name", "Expansion", "Min price", "Exemplaires en vente", "Rareté", "mkm_url", "expansion_release_date"]
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument("--start-maximized")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")

        cService = webdriver.ChromeService(executable_path=ChromeDriverManager().install())
        driver = webdriver.Chrome(service = cService, options=chrome_options)

        dico = {}
        data_lign = 0
        base_url = "https://www.cardmarket.com/en/Pokemon/Products/Singles/"
        url_extension = "https://www.cardmarket.com/en/Pokemon/Expansions/" + expansion_name + "#SetInfoSection"
        driver.get(url_extension)
        time.sleep(random.uniform(2, 5))
        release_date = driver.find_element(By.XPATH, "/html/body/main/div[2]/div[2]/div/div[1]/div[2]").text

        #On récupère le nombre de page de cartes dans l'extension choisie
        driver.get(base_url + expansion_name)
        time.sleep(random.uniform(2, 5))
        nmbr_page = int(driver.find_element(By.CLASS_NAME, "mx-1").text[-3:].replace(" ", "").replace("+", ""))
        for i in range(1, nmbr_page+1):
            time.sleep(random.uniform(2, 5))
            driver.get(base_url+ expansion_name + "?site=" + str(i))
            suffixe = "productRow"
            elements = driver.find_elements(By.XPATH, f"//*[starts-with(@id, '{suffixe}')]")
            for element in elements:
                card_elements = []

                #On récupère le nom de la carte
                card_name = element.find_elements(By.TAG_NAME, "a")[1].text
                card_elements.append(card_name)

                #On rajoute le nom de l'extension
                card_elements.append(expansion_name)

                #On récupère le prix minimum de vente
                min_price = element.find_element(By.CLASS_NAME, "col-price.pe-sm-2").text
                card_elements.append(min_price) 

                #On récupère le nombre d'exemplaires disponibles
                dispo = element.find_element(By.CLASS_NAME, "d-none.d-md-inline").text
                card_elements.append(dispo) 

                #On récupère la rareté
                rarete = element.find_element(By.TAG_NAME, "svg").get_attribute("aria-label")
                card_elements.append(rarete)


                #On récupère l'url de la carte sur mkm
                mkm_url = element.find_elements(By.TAG_NAME, "a")[1].get_attribute("href")
                card_elements.append(mkm_url)    

                #On rajoute la date de sortie de l'extension
                card_elements.append(release_date)  

                #On rajoute à l'entrée "nom de la carte" ses caractéristiques dans le dico
                dico[data_lign] = card_elements
                data_lign+=1
            
    
        data = pd.DataFrame.from_dict(dico, orient = "index", columns=titles)
        

        ###A check
        max_index = existent_data["Index"].max()
        data["Index"] = range(max_index + 1, max_index + 1 + len(data))
        new_data = pd.concat([existent_data, data], ignore_index=True, sort=False)
        new_data.to_csv('data/original_data.csv', index = False)
        driver.quit()




def add_tournament_use(indexes):
    original_data = pd.read_csv("data/original_data.csv")
    data = pd.read_csv("data/working_data.csv")
    
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")

    cService = webdriver.ChromeService(executable_path=ChromeDriverManager().install())
    driver = webdriver.Chrome(service = cService, options=chrome_options)

    base_url = "https://limitlesstcg.com/cards/"
    for index in indexes:
        lign = data[data["Index"] == index].iloc[0]
        expansion = lign["Expansion_code"]
        code = lign["Number_code"]
        url = base_url + expansion + "/" + code + "/decklists?time=1months&type=all&format=all&region=all&division=all"

        driver.get(url)
        time.sleep(random.uniform(3, 6))

        table = driver.find_element(By.CLASS_NAME, "data-table.striped")
        tournament_use = len(table.find_elements(By.TAG_NAME, "tr"))-1
        original_data.loc[index, "Tournament_last_month"] = tournament_use
    original_data.to_csv("data/original_data.csv", index=False)

    driver.quit()


def add_price_trends(indexes):
    """
    Prend en argument une liste d'index, et rajoute les tendances de prix de toutes les cartes correspondantes.
    On utilise pour cela l'URL récupéré initialement dans le tableau.
    """
    original_data = pd.read_csv("data/original_data.csv")
    data = pd.read_csv("data/working_data.csv")

    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")

    cService = webdriver.ChromeService(executable_path=ChromeDriverManager().install())
    driver = webdriver.Chrome(service = cService, options=chrome_options)

    for index in indexes:
        lign = data[data["Index"] == index].iloc[0]
        url = lign["mkm_url"]
        driver.get(url)
        time.sleep(random.uniform(3, 6))

        price_trend = driver.find_element(By.XPATH, '//*[@id="tabContent-info"]/div/div[1]/div/div[2]/dl/dd[8]/span').text
        price_30 = driver.find_element(By.XPATH, '//*[@id="tabContent-info"]/div/div[1]/div/div[2]/dl/dd[9]/span').text
        price_7 = driver.find_element(By.XPATH, '//*[@id="tabContent-info"]/div/div[1]/div/div[2]/dl/dd[10]/span').text
        original_data.loc[index, "Price trend"] = price_trend
        original_data.loc[index, "Price 7 days"] = price_7
        original_data.loc[index, "Price 30 days"] = price_30

    original_data.to_csv("data/original_data.csv", index = False)
    
    driver.quit()

