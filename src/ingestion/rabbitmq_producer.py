import pika
import json
import time
import random

def generate_user_interaction():
    """Generate a random user interaction event."""
    events = ['click', 'add_to_cart', 'purchase']
    return {
        'user_id': random.randint(1, 1000),
        'item_id': random.randint(1, 500),
        'event_type': random.choice(events),
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')
    }

def main():
    # RabbitMQ configuration
    queue_name = 'user_interactions'
    rabbitmq_host = 'localhost'

    # Connect to RabbitMQ
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=rabbitmq_host))
    channel = connection.channel()

    # Declare the queue
    channel.queue_declare(queue=queue_name)

    print(f"Producing messages to queue '{queue_name}'...")

    try:
        while True:
            # Generate a random user interaction
            interaction = generate_user_interaction()
            print(f"Sending: {interaction}")

            # Publish the interaction to the RabbitMQ queue
            channel.basic_publish(
                exchange='',
                routing_key=queue_name,
                body=json.dumps(interaction)
            )

            # Sleep for a random interval (simulate real-time events)
            time.sleep(random.uniform(0.5, 2.0))
    except KeyboardInterrupt:
        print("\nStopping producer...")
    finally:
        connection.close()

if __name__ == '__main__':
    main()