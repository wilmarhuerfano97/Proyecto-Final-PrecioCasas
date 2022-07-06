import streamlit as st
import pandas as pd
import numpy as np
import pickle

import folium
from streamlit_folium import folium_static

from sklearn.preprocessing import PolynomialFeatures

st.set_page_config(page_title='App - Pronóstico',
                    layout="wide", 
                    page_icon='🚀',  
                    initial_sidebar_state="expanded")

st.title("Pronosticando precios de casas")
st.sidebar.markdown("Características")

@st.cache
def get_data():
    return pd.read_csv('kc_house_data.csv')

data = get_data().dropna()

X = pd.DataFrame() ### dataframe vacio

banhos = st.sidebar.select_slider(
          'Número de Baños',
          options=list(sorted(set(data['bathrooms']))), value=2.5)

habitaciones = st.sidebar.number_input('Número de habitaciones', min_value=1, max_value=11, value=4, step=1)
X.loc[0,'bedrooms'] = habitaciones
X.loc[0,'bathrooms'] = banhos

area = st.sidebar.number_input('Área del inmueble', value=3180.0)
X.loc[0,'sqft_living'] = area

area_lote = st.sidebar.number_input('Área de lote', value=5438.0)
X.loc[0,'sqft_lot'] = area_lote

pisos = st.sidebar.select_slider(
          'Número de Pisos',
          options=list(sorted(set(data['floors']))), value=2.5)
X.loc[0,'floors'] = pisos

waterfront = st.sidebar.selectbox(
     '¿Vista al agua?',
     ('No', 'Sí'))

if waterfront == 'Sí': 
    waterfront = 1
else:  
    waterfront = 0
X.loc[0,'waterfront'] = waterfront

vista = st.sidebar.selectbox(
     'Puntaje de la vista',
     (0,1,2,3,4))
X.loc[0,'view'] = vista

condicion = st.sidebar.selectbox(
     'Condición del inmueble',
     (1, 2, 3, 4, 5))
X.loc[0,'condition'] = condicion

puntaje =  st.sidebar.selectbox(
     'Puntaje de construcción',
     (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13))
X.loc[0,'grade'] = puntaje

area_arriba = st.sidebar.number_input('Área sobre el sótano', value=3180)
X.loc[0,'sqft_above'] = area_arriba

area_abajo = st.sidebar.select_slider(
          'Área del sótano',
          options=list(sorted(set(data['sqft_basement']))), value=0)
X.loc[0,'sqft_basement'] = area_abajo

zipcode = st.sidebar.select_slider(
          'Codigo postal de la casa',
          options=list(sorted(set(data['zipcode']))), value=98065)
X.loc[0,'zipcode'] = zipcode

lat = st.sidebar.select_slider('Latitud', 
                        options=list(sorted(set(data['lat']))), value=47.5416)
X.loc[0,'lat'] = lat

long = st.sidebar.select_slider('Longitud',
                        options=list(sorted(set(data['long']))), value=-121.8640)
X.loc[0,'long'] = abs(long)

edad = st.sidebar.number_input('Edad', min_value=1, max_value=120, value=17, step=1)
X.loc[0,'year_old'] = edad

renovacion = st.sidebar.selectbox(
     '¿Renovación?',
     ('No', 'Sí'))

if renovacion == 'Sí': 
    renovacion = 1
else:  
    renovacion = 0

X.loc[0,'renovated_status'] = renovacion

st.markdown("""
En esta pestaña, un modelo de Machine Learning ha sido disponibilizado para generar pronósticos de precios  basado en las propuidades del inmueble. El usuario deberá suministrar las características de tal inmueble utilizando el menú de la barra izquierda. A continuación se definen la información requerida. :
     
- Número de baños: Número de baños de la propiedad a sugerir precio. Valores como 1.5 baños se refiere a la existencia de un baño con ducha y un baño sin dicha.
- Número de habitaciones: Número de habitaciones de la propiedad a sugerir precio
- Área del inmueble: Área en pies cuadrados de la propiedad a sugerir precio
- Area del lote: Área en pies destinado a la edificación.
- Número de pisos: Número de pisos de la propiedad a sugerir precio
- Vista al agua: La propiedad a sugerir precio tiene vista al agua?
- Puntaje de la vista: Puntaje de la vista de la propiedad a sugerir precio.
- Condición del inmueble: Condición general de la propiedad a sugerir precio.
- Puntaje sobre la construcción: Puntja sobre la construcción de la propiedad a sugerir precio
- Área sobre el sótano: Área en pies cuadrados de la propiedad sin contar con el sótano.
- Área del sótano: Área en pies cuadrados únicamente del espacio comprendido por el sótano.
- Código postal de la casa: Código único de cada propieda.
- Latitud: Distancia angular entre la línea ecuatorial y un punto determinado de la Tierra, dirección Norte a Sur.
- Longitud: Distancia angular entre un punto dado de la superficie terrestre y el meridiano que se toma como 0°.
- Edad de la propiedad: La antiguedad de la propiedad a sugerir precio.
- Renovación: La propiedad a sugerir precio ha sido renovada?
    """)

def model_predict(archivo, vector):
    modelo_poli1 = PolynomialFeatures(degree=2)
    modelo_poli2 = pickle.load(open(archivo, 'rb'))
    value = modelo_poli2.predict(modelo_poli1.fit_transform(vector))
    return value

if st.sidebar.button('Los parámetros han sido cargados. Calcular precio'):

    precio_v = model_predict('lin_model.sav', X)
    precio = round(abs(precio_v[0]), 2)
    st.balloons()
    st.success('El precio ha sido calculado')
    # st.write('El precio sugerido es:', )
    st.metric("Precio Sugerido", "$"+str(precio))
    
    col1, col2 = st.columns(2) ### SE DEFINEN DOS MAPAS (TWO COLUMNS: int)
    # col1 = st.columns(1) ### SE DEFINEN DOS MAPAS (TWO COLUMNS: int)

    with col1: ### first map

        st.header(f"Densidad de casas disponibles con presupuesto menor o igual al estimado: ${precio}")
        data2 = data[data['price']<=precio]
        data2['zipcode'] = data2['zipcode'].astype(str)
        
        data_aux = data2[['id','zipcode']].groupby('zipcode').count().reset_index()
        quantils = (0, 0.2, 0.4, 0.6, 0.8, 1.0)
        custom_scale = (data_aux['id'].quantile(quantils)).tolist()

        mapa = folium.Map(location=[data2['lat'].mean(),   ### avg lat
                                    data2['long'].mean()], ### avg long | (avgLAt, avgLong) map center
                          zoom_start=8) ### zoom, not big issue | visualize map | houses location

        url2 = 'https://raw.githubusercontent.com/sebmatecho/CienciaDeDatos/master/ProyectoPreciosCasas/data/KingCount.geojson'
        folium.Choropleth(geo_data=url2, ### geojson data
                          data=data_aux, ### dataframe with zipcode info
                          key_on='feature.properties.ZIPCODE', ### ZIPCODE var
                          columns=['zipcode', 'id'], ### key and value columns
                          threshold_scale=custom_scale, ### ids quantils
                          fill_color='YlOrRd', ### visual style
                          highlight=True ### cursor interaction
                         ).add_to(mapa) ### overlay map

        folium_static(mapa) ### STREAMLIT IT!
        
    col1, col2 = st.columns(2) ### SE DEFINEN DOS MAPAS (TWO COLUMNS: int)
    # col1 = st.columns(1) ### SE DEFINEN DOS MAPAS (TWO COLUMNS: int)

    with col1: ### first map

        st.header(f"Densidad de casas disponibles con presupuesto mayor al estimado: ${precio}")
        data3 = data[data['price']>precio]
        data3['zipcode'] = data3['zipcode'].astype(str)
        
        data_aux = data3[['id','zipcode']].groupby('zipcode').count().reset_index()
        quantils = (0, 0.2, 0.4, 0.6, 0.8, 1.0)
        custom_scale = (data_aux['id'].quantile(quantils)).tolist()

        mapa = folium.Map(location=[data3['lat'].mean(),   ### avg lat
                                    data3['long'].mean()], ### avg long | (avgLAt, avgLong) map center
                          zoom_start=8) ### zoom, not big issue | visualize map | houses location

        url2 = 'https://raw.githubusercontent.com/sebmatecho/CienciaDeDatos/master/ProyectoPreciosCasas/data/KingCount.geojson'
        folium.Choropleth(geo_data=url2, ### geojson data
                          data=data_aux, ### dataframe with zipcode info
                          key_on='feature.properties.ZIPCODE', ### ZIPCODE var
                          columns=['zipcode', 'id'], ### key and value columns
                          threshold_scale=custom_scale, ### ids quantils
                          fill_color='YlOrRd', ### visual style
                          highlight=True ### cursor interaction
                         ).add_to(mapa) ### overlay map

        folium_static(mapa) ### STREAMLIT IT!
        
    n_casas = data2.shape[0]
    st.header(f"Información de las {n_casas} casas relacionadas al precio estimado por el modelo:")
    att_num = data2.select_dtypes(include = ['int64','float64'])
    media = pd.DataFrame(att_num.apply(np.mean))
    mediana = pd.DataFrame(att_num.apply(np.median))
    std = pd.DataFrame(att_num.apply(np.std))
    maximo = pd.DataFrame(att_num.apply(np.max))
    minimo = pd.DataFrame(att_num.apply(np.min))
    df_EDA = pd.concat([minimo,media,mediana,maximo,std], axis = 1)
    df_EDA.columns = ['Mínimo','Media','Mediana','Máximo','std']
    # st.header('Datos descriptivos')
    df_EDA = df_EDA.drop(index =['id', 'lat', 'long','yr_built','yr_renovated'], axis=0)
    df_EDA.index =['Precio','No. Cuartos', 'No. Baños', 'Área construida (pies cuadrados)', 
                    'Área del terreno (pies cuadrados)', 'No. pisos', 'Vista agua (dummy)',
                    'Puntaje de la vista', 'Condición','Evaluación propiedad (1-13)',
                    'Área sobre tierra', 'Área sótano', 'Área construída 15 casas más próximas', 
                    'Área del terreno 15 casas más próximas']
    st.dataframe(df_EDA)
    
else:
    st.snow()
    st.error('Por favor, seleccione los parámatros de la propiedad para estimar el precio.')
