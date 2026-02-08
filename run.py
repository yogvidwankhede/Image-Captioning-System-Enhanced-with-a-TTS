"""
Quick Start Script for Image Captioning
Choose what you want to do: train, test, or run web interface
"""

import os
import sys


def print_menu():
    """Display main menu"""
    print("=" * 60)
    print("IMAGE CAPTIONING - QUICK START")
    print("=" * 60)
    print("\nWhat would you like to do?")
    print("\n1. 🎓 Train the model (requires dataset)")
    print("2. 🔍 Test inference (requires trained model)")
    print("3. 🌐 Run Streamlit web interface")
    print("4. ⚙️  Check configuration")
    print("5. 📦 Verify dataset paths")
    print("6. 🧪 Test feature extraction")
    print("7. ❌ Exit")
    print("\n" + "=" * 60)


def check_dataset():
    """Verify dataset exists"""
    from config import ANNOTATION_FILE, IMAGE_FOLDER

    print("\n📦 Checking dataset...")

    ann_exists = os.path.exists(ANNOTATION_FILE)
    img_exists = os.path.exists(IMAGE_FOLDER)

    if ann_exists:
        print(f"✅ Annotations found: {ANNOTATION_FILE}")
    else:
        print(f"❌ Annotations NOT found: {ANNOTATION_FILE}")

    if img_exists:
        print(f"✅ Images found: {IMAGE_FOLDER}")
        # Count images
        try:
            num_images = len([f for f in os.listdir(
                IMAGE_FOLDER) if f.endswith('.jpg')])
            print(f"   Total images: {num_images}")
        except:
            pass
    else:
        print(f"❌ Images NOT found: {IMAGE_FOLDER}")

    if not (ann_exists and img_exists):
        print("\n⚠️  Dataset not found!")
        print("\nTo download MS COCO 2017:")
        print("1. Training images: http://images.cocodataset.org/zips/train2017.zip")
        print("2. Annotations: http://images.cocodataset.org/annotations/annotations_trainval2017.zip")
        print("\nExtract to:")
        print("  Dataset/train2017/train2017/")
        print("  Dataset/annotations_trainval2017/annotations/")
        return False

    return True


def train():
    """Run training"""
    print("\n🎓 Starting training...")
    print("This will take several hours depending on your GPU.")

    response = input("\nContinue? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return

    import train
    train.main()


def test_inference():
    """Test inference"""
    print("\n🔍 Testing inference...")

    # Check if checkpoint exists
    from config import CHECKPOINT_DIR
    import tensorflow as tf

    latest = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    if not latest:
        print(f"❌ No checkpoint found in {CHECKPOINT_DIR}")
        print("You need to train the model first!")
        return

    print(f"✅ Found checkpoint: {latest}")

    import inference
    inference.main()


def run_streamlit():
    """Launch Streamlit interface"""
    print("\n🌐 Launching Streamlit interface...")
    print("This will open in your browser.")

    os.system("streamlit run main.py")


def show_config():
    """Display current configuration"""
    from config import print_config
    print_config()


def test_features():
    """Test feature extraction"""
    print("\n🧪 Testing feature extraction...")

    if not check_dataset():
        return

    print("\nThis will extract features from a small sample.")
    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return

    import feature_extractor
    # The module's main() will run the test


def main():
    """Main menu loop"""
    while True:
        print_menu()

        try:
            choice = input("\nEnter choice (1-7): ").strip()

            if choice == '1':
                if check_dataset():
                    train()
                else:
                    input("\nPress Enter to continue...")

            elif choice == '2':
                test_inference()
                input("\nPress Enter to continue...")

            elif choice == '3':
                run_streamlit()
                break  # Exit after Streamlit closes

            elif choice == '4':
                show_config()
                input("\nPress Enter to continue...")

            elif choice == '5':
                check_dataset()
                input("\nPress Enter to continue...")

            elif choice == '6':
                test_features()
                input("\nPress Enter to continue...")

            elif choice == '7':
                print("\nGoodbye! 👋")
                break

            else:
                print("\n❌ Invalid choice. Please enter 1-7.")
                input("Press Enter to continue...")

        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            input("Press Enter to continue...")


if __name__ == "__main__":
    main()
