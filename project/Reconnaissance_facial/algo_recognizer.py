from PIL import Image, ImageDraw, ImageFont
import face_recognition
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import encodage
import face_detection


image = 'amphi.jpeg'

#Utilisation de la fonction de détection des visages du fichier face_detection.py
face_detection.face_detection(image)

# Charger les encodages préalablement sauvegardés
with open('known_faces.pkl', 'rb') as f:
    encodage_visage_connu, nom_visage_connu = pickle.load(f)

# Dossier des images inconnues
dossier_inconnu = "people_unknown"

# Police pour le texte (optionnel, pour éviter erreur si tu veux mesurer texte)
#police = ImageFont.load_default()

present = []

for fichier in os.listdir(dossier_inconnu):
    chemin_image = os.path.join(dossier_inconnu, fichier)
    
    # Charger l'image inconnue
    image_inconnu = face_recognition.load_image_file(chemin_image)
    
    # Trouver visages et encodages
    emp_visage_inconnu = face_recognition.face_locations(image_inconnu)
    encodage_visage_inconnu = face_recognition.face_encodings(image_inconnu, emp_visage_inconnu)
    
    # Convertir en PIL pour dessin
    #image_pil = Image.fromarray(image_inconnu)
    #draw = ImageDraw.Draw(image_pil)
    
    # Pour chaque visage détecté
    for (haut, droite, bas, gauche), encodage_visage in zip(emp_visage_inconnu, encodage_visage_inconnu):
        corresp = face_recognition.compare_faces(encodage_visage_connu, encodage_visage)
        nom = "Inconnu"
        
        distances_visages = face_recognition.face_distance(encodage_visage_connu, encodage_visage)
        meilleur_indice = np.argmin(distances_visages)
        if corresp[meilleur_indice]:
            nom = nom_visage_connu[meilleur_indice]
        
        # Dessiner la boite
        #draw.rectangle(((gauche, haut), (droite, bas)), outline=(0, 0, 255), width=2)
        
        # Dessiner l'étiquette nom
        #draw.text((gauche + 6, bas - 15), nom, fill=(255, 255, 255, 255), font=police)
        if nom not in present:
            present.append(nom)
    
    # Afficher l'image annotée
    #plt.imshow(image_pil)
    #plt.axis('off')
    #plt.show()


absent=[]

for eleve in nom_visage_connu:
    if eleve not in present:
        absent.append(eleve)


#print(nom_visage_connu)
print("Les élèves présents sont :", present)
print("Les élèves absents sont :", absent)



for filename in os.listdir('people_unknown'):
    os.remove('people_unknown'+"/"+filename)