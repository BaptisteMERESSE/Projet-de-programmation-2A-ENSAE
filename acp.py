import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn

#On récupère dans un premier temps nos données en retirant toutes les cartes dont le prix minimum est inférieur à 5 centimes
original_data = pd.read_csv("original_data.csv")
original_data.to_csv("working_data.csv", index = False)

data = pd.read_csv("working_data.csv")
data = data[(data["Rareté"] != "Online Code Card") & (data["Rareté"] != "Oversized") & (data["Rareté"] != "Fixed")]
data = data.dropna(subset=["Min price"])

data["Min price"] = (data['Min price'].str[:-5] + data['Min price'].str[-4:-2]).astype(int)/100
data = data[(data["Min price"] >= 0.5)]

#On doit garder seulement les colonnes qui possèdent des valeurs numériques pour l'ACP, et retirer le symbole "euro" dans les dernières colonnes.

data["Price trend num"] = data["Price trend"].str[:-5]+"."+data["Price trend"].str[-4:-2]
data["Price 7 days num"] = data["Price 7 days"].str[:-5]+"."+data["Price 7 days"].str[-4:-2]
data["Price 30 days num"] = data["Price 30 days"].str[:-5]+"."+data["Price 30 days"].str[-4:-2]


data=data.loc[:,["Min price","Exemplaires en vente","Tournament_last_month","Price trend num","Price 7 days num","Price 30 days num"]]
print(data)
#On peut envisager deux approches. La première est de faire une analyse en composantes principales directement, 
#en laissant toutes les variables qu'on a ici pour voir ce qu'on a.
#Le problème auquel on peut penser, c'est que certaines variables vont intuitivement être très corrélées entre elles, 
#comme le prix dans les 7 derniers jours, et le prix dans les 30 derniers jours, ce qui amène à la seconde approche :
#on va retirer du tableau les colonnes "Price trend", "Price 7 days", "Price 30 days", et ensuite faire l'ACP sur les variables restantes.



#Premièrement, donc, on fait l'analyse en composantes principales sans toucher aux variables ni les normaliser.
#L'ACP vient du sous-module decomposition du module sklearn

from sklearn.decomposition import PCA
pca = PCA() #on crée l'objet qui contient l'ACP
pca.fit(data) #on ajuste l'objet sur notre tableau de données


#La première étape est de faire un tableau de la variance expliquée
tab_variance = pd.DataFrame({
        "Dimension" : [str(i + 1) for i in range(6)], 
        "pourcentage de la variance expliquée" : np.round(pca.explained_variance_ratio_ * 100),
        "Variance expliquée" : pca.explained_variance_})
tab_variance.plot.bar(x = "Dimension", y = "pourcentage de la variance expliquée")
plt.show()




#Pour finir, il peut sembler intéressant de ne réaliser l'ACP sans les cartes de l'extension nommée "151", car cette extension
#possède des cartes aux prix plus élevés en moyenne et peut donc posséder des valeurs aberrantes


