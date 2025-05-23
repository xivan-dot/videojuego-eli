# Importar librerías
import streamlit as st
import pickle
import pandas as pd
import sklearn

# Configuración de la página (debe ser la primera instrucción)
############################################################################################################################
# Título principal centrado
st.set_page_config(page_title="Modelo para la predicción de compra de videojuengos en tienda", layout="centered")

st.title("pagina para compras de video juegos en una tienda")

# montar imagen

st.image("imagen.jpg")

#Cargamos el modelo
import pickle
filename = 'modelo-reg-tree-knn-nn1.pkl'
#filename = 'modelo-reg-tree.pkl'
model_Tree,model_Knn, model_NN,variables, min_max_scaler = pickle.load(open(filename, 'rb')) #DT-Knn

#Creamos el sidebar
st.sidebar.title("datos de usuario")

def main():

    #Creamos las entradas del modelo

    def user_input_feature():
        edad = st.sidebar.number_input("Edad del cliente", 14, 52)

        option_juego = ['Mass Effect', 'Sim City', 'Dead Space', 'Battlefield', 'Fifa', 'F1', 'KOA: Reckoning']
        videojuego = st.sidebar.selectbox('Seleccione el tipo de videojuego que desea comprar', option_juego)

        option_plataforma = ['Play Station', 'PC', 'Xbox', 'Otros']
        plataforma = st.sidebar.selectbox('Plataforma en la cual funciona el videojuego', option_plataforma)

        option_sex = ['Hombre', 'Mujer']
        sexo = st.sidebar.selectbox('Sexo', option_sex)

        Consumidor_habitual = st.sidebar.checkbox('Consumido de videjuegos con mayor frecuencia', value=False)

        data = {
            "Edad": edad,
            "videojuego": videojuego,
            "Plataforma": plataforma,
            "Sexo": sexo,
            "Consumidor_habitual": Consumidor_habitual
        }

        feature = pd.DataFrame([data])  # Creamos el DataFrame correctamente
        return feature

    features = user_input_feature() #  permite ver en el front el sidebar
    

     # Preparar los datos
    data_preparada = features.copy()

    # Crear las variables dummies
    data_preparada = pd.get_dummies(data_preparada, columns=['videojuego', 'Plataforma'], drop_first = False)
    #data_preparada
    data_preparada = pd.get_dummies(data_preparada, columns=['Sexo', 'Consumidor_habitual'], drop_first = False)# se elimina una dummy porque solo tiene 2 categorias
              
    
    #Se adicionan las columnas faltantes

    data_preparada = data_preparada.reindex(columns=variables, fill_value=0)# Si falta una variable la crea y llena con ceros
    st.subheader("Datos que ingresó el cliente al modelo")
    data_preparada
    
    # Realizar predicción con NN por ser el mejor modelo recordar que hay que aplicar minmaxSacaler
    data_preparada[['Edad']]= min_max_scaler.transform(data_preparada[['Edad']])
    #st.write("datos variables del modelo con minmaxSacaler") # Revisión de aplicación de la normaliación
    #data_preparada

    if st.button('Realizar Predicción'):
        y_fut = model_NN.predict(data_preparada)
        
        st.success(f'La predicción es: {y_fut[0]:.1f} dólares') #y_fut[0]: Extrae el primer (y único) valor de la lista. :.1f: Formatea el número con una sola cifra decimal. dólares: Añade la palabra después del número.

        

if __name__ == '__main__':
    main()
