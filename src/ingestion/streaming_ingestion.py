import pika
import json
import pandas as pd
from src.utils.logger import PipelineLogger

class StreamingIngestion:
    """Handles streaming data ingestion from RabbitMQ."""

    def __init__(self, storage):
        self.storage = storage
        self.logger_name = 'streaming_ingestion'

    def ingest_streaming_data(self, queue_name, rabbitmq_host, max_messages=1000):
        """
        Ingest data from RabbitMQ and save it to the data lake.
        Args:
            queue_name (str): Name of the RabbitMQ queue.
            rabbitmq_host (str): RabbitMQ server hostname.
            max_messages (int): Maximum number of messages to consume.
        Returns:
            pd.DataFrame: DataFrame containing the ingested messages.
        """
        with PipelineLogger(self.logger_name) as logger:
            logger.info(f"Connecting to RabbitMQ queue: {queue_name}")

            # Connect to RabbitMQ
            connection = pika.BlockingConnection(pika.ConnectionParameters(host=rabbitmq_host))
            channel = connection.channel()

            # Declare the queue
            channel.queue_declare(queue=queue_name)

            messages = []

            def callback(ch, method, properties, body):
                interaction = json.loads(body)
                messages.append(interaction)
                #logger.info(f"Received message: {interaction}")

                # Stop consuming after max_messages
                if len(messages) >= max_messages:
                    channel.stop_consuming()

            # Start consuming messages
            channel.basic_consume(queue=queue_name, on_message_callback=callback, auto_ack=True)
            logger.info("Starting to consume messages...")
            channel.start_consuming()

            # Close the connection
            connection.close()

            # Convert messages to a DataFrame
            df = pd.DataFrame(messages)
            logger.info(f"Ingested {len(df)} messages from RabbitMQ")

            # Save the data to the data lake
            saved_path = self.storage.save_raw_data(
                df, source='streaming', data_type='interactions', fmt='csv'
            )
            logger.info(f"Saved streaming data to: {saved_path}")

            return df