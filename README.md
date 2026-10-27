# Détection et classification des absences de mesure de l'eye-tracker

## 1. Présentation du projet

Ce programme permet de détecter et de caractériser les instants où un eye-tracker ne fournit pas de mesure de fixation exploitable, en s'appuyant sur l'analyse de la vidéo du participant.

Il est conçu pour des enregistrements réalisés avec une **caméra fixe filmant le haut du corps du sujet**.

L'objectif est de distinguer les absences de mesure potentiellement liées au comportement du participant (tête baissée, tête tournée, déplacement, etc.) des absences survenant alors que sa posture semble normale, qui sont alors attribuées à une possible erreur de l'eye-tracker.

Le programme ajoute une colonne `type_absence` au fichier de données oculométriques afin de caractériser chaque instant.

## 2. Principe de fonctionnement

La méthode repose sur quatre étapes.

### Étape 1 — Détection des mesures manquantes

Dans le fichier de l'eye-tracker, une mesure est considérée comme manquante lorsque les coordonnées sont absentes (`NaN` ou cellule vide) ou lorsque les coordonnées valent `(0, 0)`.

Les coordonnées sont issues des colonnes correspondant aux positions du regard de l'œil gauche et de l'œil droit.

### Étape 2 — Définition d'une posture de référence

Le programme analyse la vidéo afin de détecter le visage du participant et d'en estimer la position ainsi que la taille.

Une posture de référence est calculée à partir des instants où l'eye-tracker fournit des mesures valides. Elle correspond à la posture médiane observée pendant ces instants, supposée représenter une position habituelle : participant assis, tête relativement droite et visage orienté vers l'appareil.

### Étape 3 — Classification des absences

Pour chaque instant où une mesure est manquante, le programme compare la posture observée dans la vidéo à la posture de référence.

- Si le visage conserve une position et une taille proches de la référence, l'absence est classée comme une possible erreur de l'eye-tracker.
- Si la posture s'écarte de la référence, l'absence est classée selon le comportement détecté.
- Si aucun visage n'est détecté, l'absence est classée comme `VISAGE_ABSENT`.
- Si l'instant se situe en dehors de la durée de la vidéo, il est classé comme `HORS_VIDEO`.

### Étape 4 — Enrichissement du fichier de données

Une nouvelle colonne, `type_absence`, est ajoutée au fichier de données.

Elle contient `NA` lorsque la mesure du regard est présente et une catégorie descriptive lorsque la mesure est absente.

## 3. Catégories de classification

| Valeur de `type_absence` | Description |
|---|---|
| `NA` | Mesure du regard présente. |
| `ERREUR_EYE_TRACKER` | Posture proche de la référence malgré l'absence de mesure ; absence potentiellement liée à l'appareil. |
| `TETE_BAISSEE` | Visage nettement plus bas que la position de référence. |
| `TETE_LEVEE` | Visage nettement plus haut que la position de référence. |
| `TETE_TOURNEE` | Visage de profil détecté. |
| `DEPLACEMENT_LATERAL` | Visage décalé vers la gauche ou vers la droite. |
| `RAPPROCHEMENT_CAMERA` | Visage plus grand dans l'image, suggérant un rapprochement de la caméra ou un buste penché vers l'avant. |
| `ELOIGNEMENT_CAMERA` | Visage plus petit dans l'image, suggérant un éloignement de la caméra ou un mouvement vers l'arrière. |
| `VISAGE_ABSENT` | Aucun visage détecté dans l'image, par exemple lorsque le participant sort du cadre ou qu'un objet masque son visage. |
| `HORS_VIDEO` | Instant correspondant à un temps situé en dehors de la durée de la vidéo. |

**Remarque :** ces catégories sont des interprétations fondées sur des indices visuels. Par exemple, un visage plus petit peut être dû à un mouvement du buste, mais aussi à un changement d'orientation ou de position. De même, une posture normale pendant une absence de mesure ne prouve pas à elle seule que l'eye-tracker est défectueux.

## 4. Prérequis

Le programme nécessite :

- Python 3 ;
- OpenCV pour le traitement des images et la détection du visage ;
- NumPy pour les calculs numériques ;
- pandas pour la lecture et la manipulation des données.

Matplotlib est également indiqué parmi les dépendances du programme.

### Dépendances Python

Le paquet `opencv-python` doit être installé dans une version antérieure à la version 5, car le programme utilise un détecteur de visage qui n'est plus fourni de la même manière dans OpenCV 5.

## 5. Installation

### 5.1. Créer un environnement virtuel

Depuis le répertoire du projet, exécutez :

```bash
python3 -m venv .venv
```

Activez ensuite l'environnement virtuel :

```bash
source .venv/bin/activate
```

### 5.2. Installer les dépendances

Exécutez :

```bash
python -m pip install --upgrade pip
python -m pip install "opencv-python<5" numpy pandas matplotlib
```

Ces commandes installent les bibliothèques nécessaires au fonctionnement du programme.

## 6. Utilisation

### 6.1. Exécution simple

La commande générale est :

```bash
python3 detect_absences.py video.wmv fichier_eyetracker.txt
```

Les deux arguments obligatoires sont :

- `video.wmv` : chemin vers la vidéo du participant ;
- `fichier_eyetracker.txt` : chemin vers le fichier de données de l'eye-tracker.

Exemple avec des chemins absolus :

```bash
python3 detect_absences.py \
    /chemin/vers/video.wmv \
    /chemin/vers/fichier_eyetracker.txt
```

Remplacez les chemins d'exemple par les chemins réels de vos fichiers.

### 6.2. Format du fichier eye-tracker

Par défaut, le programme attend les colonnes suivantes :

| Colonne | Description |
|---|---|
| `Capture Time` | Temps de capture, exprimé en microsecondes. |
| `Lft X Pos` | Coordonnée horizontale du regard de l'œil gauche. |
| `Lft Y Pos` | Coordonnée verticale du regard de l'œil gauche. |
| `Rt X Pos` | Coordonnée horizontale du regard de l'œil droit. |
| `Rt Y Pos` | Coordonnée verticale du regard de l'œil droit. |

Le fichier est au format texte (`.txt`) et utilise la tabulation comme séparateur.

Les noms des colonnes doivent correspondre à ceux attendus par le programme. Les valeurs temporelles sont exprimées en microsecondes.

### 6.3. Synchronisation temporelle

Le programme utilise les temps de capture du fichier eye-tracker pour déterminer les instants à examiner dans la vidéo.

La qualité de la classification dépend donc notamment de la cohérence entre les horodatages du fichier de données et ceux de la vidéo. Une synchronisation incorrecte peut conduire à comparer une mesure manquante avec une image ne correspondant pas au même instant.

## 7. Résultat attendu

Le fichier de données est enrichi avec une colonne supplémentaire nommée `type_absence`.

Cette colonne permet de distinguer les mesures présentes des différentes catégories d'absences détectées.

Exemple illustratif :

| Capture Time | Lft X Pos | Lft Y Pos | Rt X Pos | Rt Y Pos | type_absence |
|---:|---:|---:|---:|---:|---|
| 1250000 | 520 | 340 | 518 | 342 | `NA` |
| 1300000 | NaN | NaN | NaN | NaN | `ERREUR_EYE_TRACKER` |
| 1350000 | 0 | 0 | 0 | 0 | `TETE_BAISSEE` |

Les valeurs de cet exemple sont fictives et illustrent uniquement la structure attendue du résultat. La catégorie effectivement attribuée dépend de l'analyse de la vidéo et des règles de classification du programme.

## 8. Limites de la méthode

La classification repose sur la comparaison entre la posture observée et une posture de référence estimée à partir des mesures valides.

Plusieurs limites doivent être prises en compte :

- La posture de référence dépend de la qualité de la détection du visage et de la représentativité des instants où l'eye-tracker fonctionne.
- Les mouvements du sujet peuvent être difficiles à distinguer lorsque la vidéo est floue, mal éclairée ou partiellement masquée.
- Une absence de mesure peut être due à plusieurs causes simultanées, alors que le programme attribue une catégorie unique.
- Une posture visuellement normale ne permet pas de confirmer avec certitude une erreur de l'appareil.
- La méthode est conçue pour une caméra fixe filmant le haut du corps ; elle peut être moins adaptée à une caméra mobile ou à un cadrage différent.

Il est recommandé de procéder à une vérification visuelle sur un échantillon des absences classées et de comparer les catégories aux observations réelles.

## 9. Objectif scientifique

Ce programme vise à faciliter l'analyse des données oculométriques en apportant une information supplémentaire sur la nature probable des mesures manquantes.

La colonne `type_absence` permet notamment de distinguer les absences associées à un comportement visible du participant de celles qui surviennent sans changement postural évident.

Cette distinction peut contribuer à mieux interpréter les données manquantes lors des analyses statistiques ultérieures.


