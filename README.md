# Covid-colombia geográfico
La idea de este proyecto es poder visualizar casos de COVID-19 por ciudad en el mapa de Colombia. La cantidad de casos se verá reflejada en el tamaño de un círculo que reṕresenta cada ciudad


## Mapa interactivo
Asi luce el mapa interactivo para ver los casos por ciudad en toda Colombia. Pueden ingresar a este [página](http://www.charlielito.ml.s3-website.us-east-2.amazonaws.com/) para jugar con esto. Todos los dias se actualiza con los últimos datos.
![alt text][s1] 


## Casos a través del tiempo
La siguiente animación muestra como van evolucionando los casos de COVID a lo largo del territorio colombiano.
![alt text][s2] 

### Requirements
Para correr el código es necesario tener `python 3.6+` e instalar las dependencias de `requirements.txt`.
Adicionalmente necesitan setear 2 variables de entorno: `MAPBOX_TOKEN` que es un token válido para generar el mapa; `DATA_KEY` es un token válido para poder hacer query a la base de datos de [https://www.datos.gov.co/](https://www.datos.gov.co/) de donde provienen los datos de infectados.

Para generar el mapa interactivo correr:
```
python map.py --use-cache --viz --output-html index.html
```

Para generar el gif con la animación temporal de los casos correr:
```
python map_time.py --use-cache
```


[s1]: https://raw.githubusercontent.com/charlielito/covid-colombia/master/images/output.gif "S"
[s2]: https://raw.githubusercontent.com/charlielito/covid-colombia/master/images/animation.gif "S"

