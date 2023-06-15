"""
This script trains a CNN model using PyTorch Lightning for our Drone Obstacle Dataset.

The model architecture is defined in CNN_lightning.py, the data module defined in ImageDataModule.py, and the
parameters are defined in config.py.

Note that this script is HEAVILY reliant on parameters found in config.py
"""

# Import Python-native modules
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from datetime import datetime

# Import custom modules
from ImageDataModule import ImageDataModule
from CNN_lightning import CNN_lightning
from supplemental_functions import resize_image_dimensions
import config

if __name__ == "__main__":
    # Kickoff the timing of the training run
    start_time = datetime.now()
    print(f"---- Training run started at: {start_time}")
    print(f'----> Training using device: {torch.cuda.get_device_name(torch.cuda.current_device())}')

    # Set image dimensions
    image_width, image_height = resize_image_dimensions(
        image_width=config.IMAGE_WIDTH,
        image_height=config.IMAGE_HEIGHT,
        size_reduction_factor=config.SIZE_REDUCTION_FACTOR,
    )

    # Create an instance of our data module
    dm = ImageDataModule(
        data_dir=config.DATA_DIR,
        image_width=image_width,
        image_height=image_height,
        batch_size=config.BATCH_SIZE,
    )

    # Create an instance of our model
    model = CNN_lightning(
        num_dummy_images=config.NUM_DUMMY_IMAGES,
        num_channels=config.NUM_CHANNELS,
        image_width=image_width,
        image_height=image_height,
    )

    # Define the EarlyStopping callback
    early_stop_callback = EarlyStopping(
        monitor="val_loss",  # parameter to monitor
        mode="min",  # direction to monitor (ex: 'min' for loss, 'max' for acc)
        patience=config.EARLY_STOPPING_PATIENCE,  # number of epochs to wait before stopping
        verbose=True,  # log information to the terminal
        min_delta=config.MIN_DELTA,  # minimum change in monitored value to qualify as improvement
    )

    # Create an instance of our trainer, and train the model
    trainer = pl.Trainer(
        max_epochs=config.MAX_EPOCHS,
        min_epochs=config.MIN_EPOCHS,
        accelerator=config.ACCELERATOR,
        devices=config.DEVICES,
        callbacks=[early_stop_callback],
        log_every_n_steps=config.LOG_EVERY_N_STEPS,
    )
    print(f">>>> Training model for a max {config.MAX_EPOCHS} epochs")
    trainer.fit(model, datamodule=dm)

    # End the timing of the training run
    end_time = datetime.now()
    print(f"---- Training run ended at: {end_time}")
    print(f"---- Training run duration: {end_time - start_time}")

    if config.ALSO_TEST:
        # Test the model

        # Kickoff the timing of the testing run
        start_time = datetime.now()
        print(f"---- Testing run started at: {start_time}")
        trainer.test(model, datamodule=dm)

        # End the timing of the testing run
        end_time = datetime.now()
        print(f"---- Testing run ended at: {end_time}")
        print(f"---- Testing run duration: {end_time - start_time}")

    if config.SAVE_MODEL:
        # Save model
        torch_model_filename = config.TORCH_MODEL_FILENAME
        print(f">>>> Saving model as {torch_model_filename}")
        torch.save(model.state_dict(), torch_model_filename)

    """
    # Example code to load (from a '.pt' file) and test model
    start_time = datetime.now()
    print(f'Testing run started at: {start_time}')

    # Load model
    model = CNN_lightning(num_channels=num_channels,
                image_width=image_width,
                image_height=image_height)
    print(f'Load model: {torch_model_filename}')
    model.load_state_dict(torch.load(torch_model_filename))
    model.eval()

    # Test model
    test_result = trainer.test(model, datamodule=dm)

    end_time = datetime.now()
    print(f'Testing run ended at: {end_time}')
    print(f'Testing run duration: {end_time - start_time}')
    """
