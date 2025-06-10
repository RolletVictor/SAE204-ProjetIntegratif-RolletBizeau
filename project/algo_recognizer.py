from PIL import Image, ImageDraw, ImageFont
import face_recognition
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle


# Charger les encodages préalablement sauvegardés
with open('known_faces.pkl', 'rb') as f:
    encodage_visage_connu, nom_visage_connu = pickle.load(f)

'''encodage_visage_connu = []
nom_visage_connu = []


for ligne in table_alt:
    chemin = ligne[2]
    
    # Charger l'image connue
    image = face_recognition.load_image_file(chemin)
    
    # Extraire l'encodage du visage (on prend le premier visage détecté)
    encodages = face_recognition.face_encodings(image)
    if len(encodages) == 0:
        print(f"Aucun visage trouvé dans la base de données, image ignorée.")
    
    encodage_visage = encodages[0]
    encodage_visage_connu.append(encodage_visage)
    
    # Extraire un nom à partir du nom de fichier (par exemple sans extension)
    nom_visage_connu.append(ligne[0]+" "+ligne[1])

print(f"{len(encodage_visage_connu)} visages connus chargés.")'''

# Dossier des images inconnues
dossier_inconnu = "people_unknown"

# Police pour le texte (optionnel, pour éviter erreur si tu veux mesurer texte)
police = ImageFont.load_default()

present = []

for fichier in os.listdir(dossier_inconnu):
    chemin_image = os.path.join(dossier_inconnu, fichier)
    
    # Charger l'image inconnue
    image_inconnu = face_recognition.load_image_file(chemin_image)
    
    # Trouver visages et encodages
    emp_visage_inconnu = face_recognition.face_locations(image_inconnu)
    encodage_visage_inconnu = face_recognition.face_encodings(image_inconnu, emp_visage_inconnu)
    
    # Convertir en PIL pour dessin
    image_pil = Image.fromarray(image_inconnu)
    draw = ImageDraw.Draw(image_pil)
    
    # Pour chaque visage détecté
    for (haut, droite, bas, gauche), encodage_visage in zip(emp_visage_inconnu, encodage_visage_inconnu):
        corresp = face_recognition.compare_faces(encodage_visage_connu, encodage_visage)
        nom = "Inconnu"
        
        distances_visages = face_recognition.face_distance(encodage_visage_connu, encodage_visage)
        meilleur_indice = np.argmin(distances_visages)
        if corresp[meilleur_indice]:
            nom = nom_visage_connu[meilleur_indice]
        
        # Dessiner la boite
        draw.rectangle(((gauche, haut), (droite, bas)), outline=(0, 0, 255), width=2)
        
        # Dessiner l'étiquette nom
        draw.text((gauche + 6, bas - 15), nom, fill=(255, 255, 255, 255), font=police)
        present.append(nom)
    
    # Afficher l'image annotée
    plt.imshow(image_pil)
    plt.axis('off')
    plt.show()
    
    # Optionnel: enregistrer l'image avec un nouveau nom
    # image_pil.save(f"result_{fichier}")

absent=[]

for eleve in nom_visage_connu:
    if eleve not in present:
        absent.append(eleve)


#print(nom_visage_connu)
print("Les éléves présents sont :", present)
print("Les éléves absents sont :", absent)