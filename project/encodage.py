import csv
import face_recognition
import pickle
import os

encodings = []
names = []

# Lecture du fichier CSV
with open('bdd.csv', newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)

    for row in reader:
        nom = row['nom']
        prenom = row['prenom']
        chemin = row['image']

        if os.path.exists(chemin):
            try:
                image = face_recognition.load_image_file(chemin)
                face_encs = face_recognition.face_encodings(image)

                if face_encs:
                    encodings.append(face_encs[0])
                    names.append(nom+" "+prenom)
                    print(f"[INFO] Encodage réussi pour {nom} {prenom}")
                else:
                    print(f"[WARNING] Aucun visage détecté dans {chemin}")
            except Exception as e:
                print(f"[ERROR] Problème avec l'image {chemin} : {e}")
        else:
            print(f"[ERROR] Chemin invalide ou image introuvable : {chemin}")

# Sauvegarde dans un fichier pickle
with open('known_faces.pkl', 'wb') as f:
    pickle.dump((encodings, names), f)

print(f"[INFO] {len(encodings)} visages encodés et sauvegardés dans 'known_faces.pkl'")

