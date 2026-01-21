import pika
import json

def process_interaction(interaction):
    """Process a user interaction event."""
    print(f"Processing interaction: {interaction}")

def main():
    # RabbitMQ configuration
    queue_name = 'user_interactions'
    rabbitmq_host = 'localhost'

    # Connect to RabbitMQ
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=rabbitmq_host))
    channel = connection.channel()

    # Declare the queue
    channel.queue_declare(queue=queue_name)

    print(f"Consuming messages from queue '{queue_name}'...")

    # Define the callback function for processing messages
    def callback(ch, method, properties, body):
        interaction = json.loads(body)
        process_interaction(interaction)

    # Start consuming messages
    channel.basic_consume(queue=queue_name, on_message_callback=callback, auto_ack=True)

    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        print("\nStopping consumer...")
    finally:
        connection.close()

if __name__ == '__main__':
    main()