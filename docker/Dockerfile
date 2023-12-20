FROM python:3.10 as api
RUN \
    pip install --upgrade pip && \
    pip install --upgrade setuptools && \
    pip install --no-cache-dir Cython==0.29.33
COPY docker/requirements_api.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt &&  \
    rm /requirements.txt
RUN pip install -U huggingface_hub
RUN mkdir -p "/opt/rec_system"
COPY data/products.csv /opt/rec_system/data/products.csv
COPY data/embeddings/* /opt/rec_system/data/embeddings/
COPY src/api/back.py /opt/rec_system/src/api/back.py
WORKDIR "/opt/rec_system/src/api"
ENTRYPOINT ["python", "back.py"]

FROM python:3.10 as recommendations
RUN \
    pip install --upgrade pip && \
    pip install --upgrade setuptools
COPY docker/requirements_recommendations.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt &&  \
    rm /requirements.txt
RUN mkdir -p "/opt/rec_system"
COPY data/images/images_1/* /opt/rec_system/data/images/images_1/
COPY data/images/images_2/* /opt/rec_system/data/images/images_2/
COPY data/images/images_3/* /opt/rec_system/data/images/images_3/
COPY data/service_images/* /opt/rec_system/data/service_images/
COPY data/web_images/* /opt/rec_system/data/web_images/
COPY data/product_images.csv /opt/rec_system/data/product_images.csv
COPY data/products.csv /opt/rec_system/data/products.csv
COPY src/recommendations/* /opt/rec_system/src/web/
WORKDIR "/opt/rec_system"
CMD streamlit run src/web/app.py --server.port 81

FROM python:3.10 as search
RUN \
    pip install --upgrade pip && \
    pip install --upgrade setuptools
COPY docker/requirements_search.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt &&  \
    rm /requirements.txt
RUN mkdir -p "/opt/search_system"
COPY data/images/images_1/* /opt/search_system/data/images/images_1/
COPY data/images/images_2/* /opt/search_system/data/images/images_2/
COPY data/images/images_3/* /opt/search_system/data/images/images_3/
COPY data/service_images/* /opt/search_system/data/service_images/
COPY data/web_images/* /opt/search_system/data/web_images/
COPY data/product_images.csv /opt/search_system/data/product_images.csv
COPY data/products.csv /opt/search_system/data/products.csv
COPY src/search/* /opt/search_system/src/web/
WORKDIR "/opt/search_system"
CMD streamlit run src/web/app.py --server.port 80
