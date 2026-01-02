FROM tensorflow/serving:latest

COPY ./output /models
COPY ./config /model_config

ENV MONITORING_CONFIG /model_config/prometheus.config
ENV MODEL_BASE_PATH /models
ENV MODEL_NAME serving_model

COPY tf_serving_entrypoint.sh /usr/bin/tf_serving_entrypoint.sh

RUN chmod +x /usr/bin/tf_serving_entrypoint.sh

ENTRYPOINT []

CMD ["/usr/bin/tf_serving_entrypoint.sh"]