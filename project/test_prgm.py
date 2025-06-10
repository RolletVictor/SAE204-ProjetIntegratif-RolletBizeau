import csv
import pygame

table_alt = []
with open ('bdd.csv', newline = "") as csvfile:
    reader = csv.reader(csvfile, delimiter=",")
    for row in reader:
        table_alt.append(row)

liste_nom = []
liste_visage = []

for ligne in table_alt:
    liste_nom.append(ligne[0])
    liste_visage.append(ligne[2])


pygame.init()
window_resolution = (200,200)

pygame.display.set_caption("Python #37")
window_surface = pygame.display.set_mode(window_resolution)

image = pygame.image.load(liste_visage[5])
image.convert()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            launched = False

    window_surface.blit(image, [10,10])
    pygame.display.flip()



#print(liste_visage,liste_nom)