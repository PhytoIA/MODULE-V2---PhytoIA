import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from numpy.random import default_rng as rng

#Definition
potato_late_confirmed = False

#Page configuration
st.set_page_config(page_title="PhyoIA Système", layout="wide")
img = Image.open("logo-icon.png")
st.set_page_config(page_icon=img)

# def model_prediction(test_image):
#     model = tf.keras.models.load_model('trained_model.keras')
#     image = tf.keras.preprocessing.image.load_img(test_image, target_size=(224,224))
#     input_arr = tf.keras.preprocessing.image.img_to_array(image)
#     input_arr = np.array([input_arr]) # Convert single image to a batch
#     prediction = model.predict(input_arr)
#     result_index = np.argmax(prediction)
#     return result_index

#Tensorflow Model Prediction
def model_prediction(test_image):
    interpreter = tf.lite.Interpreter(model_path='model.20_finedtuned_tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(224,224))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    interpreter.set_tensor(input_details[0]['index'],input_arr)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    top3 = np.argsort(prediction[0])[-3:][::-1] # Ordre croissant de toutes les prédictions (insérés dans le premier [] de l'image), sélections des trois plus grands (derniers de la liste) et mise en ordre décroissant.
    result_index = np.argmax(prediction)
    return result_index, top3, prediction


#Sidebar
st.sidebar.title("Système multimodal 👨‍🌾​")
    # image_path = "plant_image.png"
    # st.image(image_path, width="stretch")
app_mode = st.sidebar.selectbox("Sélection Page", [ "Gestion - Reconnaissance Maladies", "Gestion - Prévision climatique","Gestion - État édaphique", "Gestion - Alertes", "Information", "À propos",])
st.logo("logo-nav.png", size='large')
st.sidebar.link_button(url = "https://phytoia.github.io/PhytoIA---Systeme-multimodal-agricole-environnemental-/", label="🏠 Retourner à l'accueil", width="stretch")

# st.sidebar.page_link("https://phytoia.github.io/PhytoIA---Systeme-multimodal-agricole-environnemental-/", label="🏠 Retourner à l'accueil")

# st.sidebar.link_button("Retourner à l'accueil ", "https://phytoia.github.io/PhytoIA---Systeme-multimodal-agricole-environnemental-/", icon='deconnexion-modified.png')

#Style - HTML/CSS Personnalisation
css = """
.st-key-treatment{
    background: linear-gradient(135deg,rgba(255,107,53,0.08) 0%,#0A0F1E 60%);
}
"""
st.html(f"<style>{css}</style>")

#Home Page
if(app_mode=="Information"):
    st.header("SYSTÈME INTELLIGENT GESTIONNEL DE MALADIES CULTURE")
    st.markdown("""
    Bienvenue sur la Plateforme d'Aide Décisionnel Agricole des Maladies! 🌿🔍 
    
    Avec PhytoIA, notre mission est de concevoir une plateforme favorisant l'aide à la prise de décision agricole en combinant la détection des maladies des cultures par intelligence artificielle, une multitude de données environnementales pertinentes, l'anticipation des risques de pertes et la mise en oeuvre de stratégies d'intervention adaptées, économiques et durables.
    """)

    # with st.container(border=True, width="stretch"):
    #     st.markdown("""#### Résumé - Météo actuelle ⛅​""")
    #     st.write("""### Précipitations:""")
    #     st.write("""### Humidité relative:""")
    #     st.write("""### Température:""")
    #     st.write("""### Vents:""")
    #     st.write("""### Ciel:""")
    # METRICS

    st.markdown("""    
    Cette application web interactive répond ainsi à notre objectif inital en permettant aux producteurs de surveiller toutes conditions agroclimatiques et édaphiques des cultures, d'analyser leur état grâce à un modèle d'IA précis et d'obtenir des alertes ainsi que des recommandations personnalisées. Ensemble, protégons nos cultures pour assurer des récoltes saines et productives!
    
    ### Fonctionnement
    1. **Prévision agroclimatique:** renseignez-vous sur les conditions environnementales clées qui sont à venir et qui varient  dans votre secteur L'anticipation d'émergence potentielle de maladies végétales des cultures, favorisées par ces facteurs, est analysée et des alertes rétroactives vous sont offertes au moment opportun. 
                
    2. **État du sol:** s'occuper de l'édaphologie des plantes est crucial afin de maintenir les sols, étant leur habitat naturel, en pleine santé. Tenez à jour la gestion de toutes propriétés de vos sols.
                
    3. **Détection maladies:** la détection réactive et rapide de symptômes phytopatologiques permet de retracer les maladies des cultures afin d'agir efficacement. 
                
    4. **Résultats:** observez les résultats et recommendations concrètes porposés à chaque étape logistique de la procédure pour anticiper les risques agricoles liés aux maladies des cultures. Des mesures préventives et curatives (traitements personnalisés) ainsi que des explications scientifiques oriente la prise de décision.

    ### Pourquoi nous choisir?
    - **Précision:** Notre système utilise des méthodes de ML (apprentissage de machine) structurés et basés sur un vaste ensemble de données.
                
    - **Utilisation conviviale:** interface simple et intuitive pour une gestion des cultures adéquate. 
    - **Rapide et efficace:** Receive results in seconds, allowing for quick decision-making.

    ### Débutez
    Glissez à travers la page de navigation sur le côté pour  découvrir et suivre chaque fonctionnalité ou outil technique. En cette ère d'aléas climatiques fréquents, améliorons l'anticipation et la gestion des maladies des cultures en réduisant l'incertitude!

    ### À propos
    Apprenez plus à propos du projet, de sa conception et autres informations en rapport avec le modèle en retournant sur la page d'accueil. 
    """)

#About page
elif(app_mode=="À propos"):
    st.header("About")
    st.markdown("#### Modèle: EfficientNetB0 fine-tuned")
    st.markdown("""
    #### About Dataset
    This dataset is recreaxted using offline augmentation from the original dataset. The original dataset can be found on this github repo. This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes. The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure. A new directory containing 33 test images is created later for prediction purpose.
    #### Content
    1. Train (70925 images)
    2. Valid (17572 images)
    3. Test (33 images)
    """)

#Prediction Page
elif(app_mode=="Gestion - Reconnaissance Maladies"):
    st.header("Reconnaissance Maladies 🔍")
    test_image = st.file_uploader("Choisissez une image (**claire**):")

    #Affichage button
    if(st.button("Afficher image  ​📷​", width="stretch")):
        col1_1, col2_1, col3_1 = st.columns(3)
        with col2_1:
            st.image(test_image, width=1000)

    #Filtre
    potato_confirmed= False
    tomato_confirmed= False
    filtreculture_mode = st.selectbox("Spécifier culture en cas d'ambiguïté (surtout mildiou):", ["Aucune", "Pomme de terre", "Tomate"], width="stretch")
    if (filtreculture_mode=="Pomme de terre"):             
        potato_confirmed = True

    if (filtreculture_mode=="Tomate"):             
        tomato_confirmed = True

    #Predict button
    if(st.button("Analyse (Prédire)  🌱", width="stretch")):
        with st.spinner("Attendez un moment.."):
            st.write("Résultats interprétés:")
            result_index, top3, prediction = model_prediction(test_image)
            #Define Class
            class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
            class_name_result = class_name[result_index]
            if class_name_result == 'Tomato___Late_blight' and potato_confirmed:
                st.success("✅​ Le modèle prédit que c'est: Potato___Late_blight")
                st.info("ℹ️​ Notez que le mildiou de la tomate et mildiou de la pomme de terre sont similaires. Il s'agit du même problème: une maladie fongique se développant par un microbe de type Phytophthora infestans. Le modèle a identifié un mildiou avec une forte confiance. Comme la culture sélectionnée correspond à la pomme de terre, le diagnostic est adapté à cette culture.")
                potato_late_confirmed = True
            
            elif class_name == 'Potato___Late_blight' and tomato_confirmed:
                st.success("✅​ Le modèle prédit que c'est: Tomato___Late_blight")
                st.info("ℹ️​ Notez que le mildiou de la tomate et mildiou de la pomme de terre sont similaires. Il s'agit du même problème: une maladie fongique se développant par un microbe de type Phytophthora infestans. Le modèle a identifié un mildiou avec une forte confiance. Comme la culture sélectionnée correspond à la tomate, le diagnostic est adapté à cette culture.")
                #  \n\n --> Space lines

            else:
                counter = 1
                for i in top3[:1]:
                    prediction_pourcentage = int(prediction[0][i] * 100)
                    st.success("Le modèle prédit que c'est: {}".format(class_name[result_index]) + f" à {prediction[0][i] * 100:.2f} %")
                    st.progress(prediction_pourcentage, 'Pourcentage niveau de confiance:', width="stretch")
                for i in top3[1:]:
                    st.warning(f'{counter + 1}. {class_name[i]} à {prediction[0][i] * 100:.2f} %') # Nom et niveau de confiance en fonction de la classe
                    prediction_pourcentage = int(prediction[0][i] * 100)
                    st.progress(prediction_pourcentage, 'Pourcentage niveau de confiance:', width="stretch")
                    counter += 1
            
            # Description et information personnalisée
            st.markdown("---")

            if class_name_result == 'Potato___Late_blight' or potato_late_confirmed:
                with st.container(border=True):
                    st.markdown("### 🍃​ Description")
                    st.write("Le mildiou de la pomme de terre est une maladie cryptogamique (causée par des micro-organismes de type oomycète) dévastatrice pour de nombreuses cultures. Elle se manifeste par des taches décolorées ou huileuses sur les feuilles, un duvet blanc (mycélium) sur la face inférieure, et le pourrissement rapide des tiges et des fruits.")
                with st.container(border=True):
                    st.markdown("### ​🔍​​ Symptômes")
                    st.write("Les premiers symptômes sont l'apparition de taches noires (vertes foncées) à l'extrémité des feuilles et sur les tiges pouvant avoir une bordure jaunâtre et verdâtre. Une moisissure blanche (mycélium ou filament fongique de hyphe) peut apparaître du côté inférieur des feuilles par temps humide ou pluvieux. Les tubercules infectés développent des lésions, des taches noires et grisâtres s’élargissant qui virent au brun-rougeâtre sous la peau.")
                col1_2, col2_2 = st.columns(2)
                with col1_2:
                    with st.container(border=True, height=300):
                        st.markdown("### 🌤️​​ Causes")
                        st.write("Le mildiou est une maladie végétale se développant à cause de facteurs environnementaux tels qu'une **haute humidité** et une **température modérée** (18-24 degrés). In id cursus mi pretium tellus duis convallis. Tempus leo eu aenean sed diam urna tempor. Pulvinar vivamus fringilla lacus nec metus bibendum egestas. Iaculis massa nisl malesuada lacinia integer nunc posuere.")
                with col2_2:   
                    with st.container(border=True, height=300):
                        st.markdown("### 🦠​​ Méthode Propagation")
                        st.write("Les premiers symptômes sont l'apparition de taches noires (vertes foncées) à l'extrémité des feuilles et sur les tiges pouvant avoir une bordure jaunâtre et verdâtre. Une moisissure blanche (mycélium ou filament fongique de hyphe) peut apparaître du côté inférieur des feuilles par temps humide ou pluvieux. Les tubercules infectés développent des lésions, des taches noires et grisâtres s’élargissant qui virent au brun-rougeâtre sous la peau.")
                st.markdown("---")
                with st.container(border=True):
                    st.markdown("### 🌿 Traitements")
                    st.write("Plusieurs traitements sont proposées en lien avec cette maladie. Lorem ipsum dolor sit amet consectetur adipiscing elit. Quisque faucibus ex sapien vitae pellentesque sem placerat. In id cursus mi pretium tellus duis convallis. Tempus leo eu aenean sed diam urna tempor. Pulvinar vivamus fringilla lacus nec metus bibendum egestas. Iaculis massa nisl malesuada lacinia integer nunc posuere.")
               
                    tab1, tab2, tab3 = st.tabs(["🧺​ Traitements", "🌡️​ Facteurs favorables", "🚨​ Risque"])

                    #Tab 1
                    tab1.subheader("Recommendations proposés")

                    #Tab 2
                    tab2.subheader("Conditions climatiques (7-14 jours passé)")
                    with tab2:
                        a, b = st.columns(2)
                        c, d = st.columns(2)

                        a.metric("Température", "30°C", "-9°C", border=True)
                        b.metric("Vent", "4 kmh", "2 kmh", border=True)

                        c.metric("Humidité", "77%", "5%", border=True)
                        d.metric("Précipitations", "20 mm", "-2 mm", border=True)

                    #Tab 3
                    tab3.subheader("Liste des risques")

                        # with tab1:
                        #     box1 = st.empty() 
                        #     with box1.container(border=True, width='stretch', height=150):
                        #         st.markdown('###### 1. First treatment')
                        #         if st.button('Compléter', width="stretch", key='button_treat_1'):
                        #             box1.empty()                

                        # # 1. Initialize memory: Start with the box hidden (False)
                        # if "show_box" not in st.session_state:
                        #     st.session_state.show_box = False

                        # # 2. The Main Action Button: Click this to start the process
                        # if st.button("Run Main Action"):
                        #     # This turns the box ON every time you click it
                        #     st.session_state.show_box = True
                        #     st.rerun()

                        # # 3. The Conditional Box: Only shows up when an action triggers it
                        # if st.session_state.show_box:
                        #     with st.container(border=True):
                        #         st.warning("⚠️ Warning: Are you sure you want to proceed?")
                        #         st.write("This action might overwrite your current data.")
                                
                        #         # Put the decline button INSIDE the box
                        #         if st.button("❌ Decline & Cancel"):
                        #             # This turns the box OFF when clicked
                        #             st.session_state.show_box = False
                        #             st.rerun()
                                    
                        #         if st.button("✅ Confirm Action"):
                        #             st.success("Action processed successfully!")
                        #             # Turn off the box after a successful confirmation
                        #             st.session_state.show_box = False
                        #             st.rerun()



                    # Add metrics, (IF CONDITIONS: if humidity high: this treatment)



                    #Filtre: réduit les ambiguïtés entre maladies visuellement similaires. Le modèle effectue une classification visuelle indépendante. Ensuite, un filtre contextuel basé sur la culture sélectionnée est appliqué afin de résoudre les ambiguïtés entre les classes de mildiou de la tomate et de la pomme de terre, causées par le même pathogène (Phytophthora infestans).

                    #Maladie détectée :
                    # 🍂 Potato Late Blight

                    # Confiance :
                    # 96.4 %

                    # Informations (description, etc) :
                    # ...

                    # Facteurs favorables :
                    # ...

                    # → Voir l'analyse environnementale
