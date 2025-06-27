import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def explorer_structure_donnees(chemin_fichier):
    """
    Explore la structure réelle des données pour comprendre le format.
    """
    try:
        data = np.load(chemin_fichier, allow_pickle=True)
    except FileNotFoundError:
        print(f"❌ Fichier introuvable: {chemin_fichier}")
        return None
    
    print("🔍 EXPLORATION DE LA STRUCTURE")
    print("=" * 50)
    print(f"Fichier: {os.path.basename(chemin_fichier)}")
    print(f"Nombre d'éléments principaux: {len(data)}")
    
    for i, element in enumerate(data):
        print(f"\n📊 Élément {i}:")
        print(f"   Type: {type(element)}")
        if hasattr(element, 'shape'):
            print(f"   Shape: {element.shape}")
        if hasattr(element, 'dtype'):
            print(f"   Dtype: {element.dtype}")
        
        # Si c'est un array, explorer plus en détail
        if isinstance(element, np.ndarray):
            if element.dtype == 'object':
                print(f"   ⚠️  Array d'objets détecté")
                print(f"   Premiers éléments (jusqu'à 5):")
                for j in range(min(5, len(element.flat))):
                    sub_element = element.flat[j]
                    if hasattr(sub_element, 'shape'):
                        print(f"      [{j}] Shape: {sub_element.shape}, Type: {type(sub_element)}")
                    else:
                        print(f"      [{j}] Type: {type(sub_element)}, Valeur: {str(sub_element)[:50]}...")
            else:
                print(f"   📈 Array numérique standard")
                if element.size > 0:
                    print(f"   Min: {element.min():.3f}, Max: {element.max():.3f}")
                    print(f"   Aperçu: {element.flat[:5]}...")
                else:
                    print("   Array vide.")
    
    return data

def plot_clusters_significatifs_simple(chemin_fichier):
    """
    Génère un plot simple des zones significatives et le sauvegarde.
    """
    data = np.load(chemin_fichier, allow_pickle=True)
    
    n_elements = len(data)
    # Adapte le nombre de sous-plots en fonction du nombre d'éléments
    ncols = 2
    nrows = (n_elements + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows), squeeze=False)
    axes = axes.flatten()
    
    fig.suptitle(f'Clusters Significatifs - {os.path.basename(chemin_fichier)}', fontsize=16, fontweight='bold')
    
    for i, element in enumerate(data):
        ax = axes[i]
        
        try:
            matrice = None
            titre_plot = f'Cluster {i+1}'

            # Cas 1: Array numérique standard
            if isinstance(element, np.ndarray) and element.dtype != 'object' and element.ndim == 2:
                matrice = element
                    
            # Cas 2: Array d'objets - essayer de trouver la première matrice
            elif isinstance(element, np.ndarray) and element.dtype == 'object':
                for j, sub_elem in enumerate(element.flat):
                    if isinstance(sub_elem, np.ndarray) and sub_elem.ndim == 2:
                        matrice = sub_elem
                        titre_plot = f'Cluster {i+1} (matrice interne {j})'
                        print(f"✅ Cluster {i+1}: Matrice {matrice.shape} trouvée à l'index interne {j}")
                        break
            
            if matrice is not None:
                # Plot de la matrice
                vmax = np.abs(matrice).max()
                im = ax.imshow(matrice.T, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
                ax.set_xlabel('Temps (échantillons)')
                ax.set_ylabel('Électrodes')
                plt.colorbar(im, ax=ax, label='Amplitude')
            else:
                ax.text(0.5, 0.5, 'Structure non reconnue\npour le plotting', ha='center', va='center', transform=ax.transAxes)

        except Exception as e:
            ax.text(0.5, 0.5, f'Erreur: {str(e)[:50]}...', ha='center', va='center', transform=ax.transAxes)
        
        ax.set_title(titre_plot)

    # Masquer les axes inutilisés
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # --- SAUVEGARDE DE LA FIGURE ---
    base_name = os.path.splitext(os.path.basename(chemin_fichier))[0]
    output_filename = f"{base_name}_simple_plot.png"
    # os.getcwd() retourne le répertoire où le script est lancé
    output_path = os.path.join(os.getcwd(), output_filename)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot simple sauvegardé : {output_path}")
    plt.close(fig) # Ferme la figure pour libérer la mémoire

def plot_zones_significatives_seuil(chemin_fichier, seuil=2.0):
    """
    Génère un plot des zones dépassant un seuil et le sauvegarde.
    """
    data = np.load(chemin_fichier, allow_pickle=True)
    
    n_elements = len(data)
    ncols = 2
    nrows = (n_elements + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 6 * nrows), squeeze=False)
    axes = axes.flatten()
    
    fig.suptitle(f'Zones Significatives (seuil > {seuil}) - {os.path.basename(chemin_fichier)}', fontsize=16, fontweight='bold')
    
    for i, element in enumerate(data):
        ax = axes[i]
        
        try:
            matrice = None
            # Chercher une matrice 2D à analyser dans les données
            if isinstance(element, np.ndarray) and element.dtype != 'object' and element.ndim == 2:
                matrice = element
            elif isinstance(element, np.ndarray) and element.dtype == 'object':
                for sub_elem in element.flat:
                    if isinstance(sub_elem, np.ndarray) and sub_elem.ndim == 2:
                        matrice = sub_elem
                        break
            
            if matrice is not None:
                masque_signif = np.abs(matrice) > seuil
                affichage = np.zeros_like(matrice)
                affichage[masque_signif] = matrice[masque_signif]
                
                vmax = np.abs(matrice).max()
                im = ax.imshow(affichage.T, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
                
                n_signif = np.sum(masque_signif)
                pourcentage = (n_signif / matrice.size) * 100 if matrice.size > 0 else 0
                
                ax.set_title(f'Cluster {i+1}\n{n_signif} points sig. ({pourcentage:.1f}%)')
                ax.set_xlabel('Temps (échantillons)')
                ax.set_ylabel('Électrodes')
                plt.colorbar(im, ax=ax, label='Amplitude')
                print(f"📊 Cluster {i+1}: {n_signif}/{matrice.size} points significatifs ({pourcentage:.1f}%)")
            else:
                ax.set_title(f'Cluster {i+1}')
                ax.text(0.5, 0.5, 'Pas de matrice 2D trouvée', ha='center', va='center', transform=ax.transAxes)
                
        except Exception as e:
            ax.set_title(f'Cluster {i+1} - Erreur')
            ax.text(0.5, 0.5, f'Erreur: {str(e)[:30]}...', ha='center', va='center', transform=ax.transAxes)

    # Masquer les axes inutilisés
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # --- SAUVEGARDE DE LA FIGURE ---
    base_name = os.path.splitext(os.path.basename(chemin_fichier))[0]
    seuil_str = str(seuil).replace('.', '_') # Pour un nom de fichier valide
    output_filename = f"{base_name}_seuil_{seuil_str}_plot.png"
    output_path = os.path.join(os.getcwd(), output_filename)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot de seuil sauvegardé : {output_path}")
    plt.close(fig)

def main():
    """Interface principale pour l'exploration et la visualisation."""
    if len(sys.argv) > 1:
        chemin_fichier = ' '.join(sys.argv[1:])
    else:
        chemin_fichier = input("📂 Entrez le chemin vers le fichier .npy: ").strip().strip("'\"")

    if not os.path.exists(chemin_fichier):
        print(f"❌ Fichier introuvable: {chemin_fichier}")
        return
    
    print("\n--- Exploration de la structure des données ---")
    explorer_structure_donnees(chemin_fichier)
    
    while True:
        print("\n" + "="*50)
        choix = input(
            "Que voulez-vous faire?\n"
            "  1️⃣  Générer et sauvegarder le plot simple\n"
            "  2️⃣  Générer et sauvegarder le plot des zones seuillées\n"
            "  3️⃣  Les deux\n"
            "  q   Quitter\n"
            "Votre choix: "
        ).strip().lower()
        
        if choix in ['1', '3']:
            print("\n🎨 Génération du plot simple...")
            plot_clusters_significatifs_simple(chemin_fichier)
        
        if choix in ['2', '3']:
            try:
                seuil_input = input("🎯 Entrez un seuil de significativité (défaut 2.0): ")
                seuil = float(seuil_input) if seuil_input else 2.0
            except ValueError:
                seuil = 2.0
                print("⚠️ Entrée invalide, utilisation du seuil par défaut 2.0")
            print(f"\n🎨 Génération du plot des zones significatives (seuil > {seuil})...")
            plot_zones_significatives_seuil(chemin_fichier, seuil)
        
        if choix == 'q':
            break
        elif choix not in ['1', '2', '3']:
            print("❌ Choix invalide, veuillez réessayer.")

    print("\n✅ Terminé!")

if __name__ == "__main__":
    main()