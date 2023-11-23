import torch
from tqdm import tqdm # progress bar

# Training loop (contains forward and backward pass, with updates of the parameters)
def train_loop(dataloader, model, optimizer, loss_function, epoch, num_epochs, device):
    model.train()
    
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    batch_size = dataloader.batch_size

    train_loss, correct_predictions = 0, 0

    with tqdm(dataloader, total=num_batches, unit='step', disable=False) as tepoch:
        tepoch.set_description(f'Epoch {epoch}/{num_epochs}')

        for batch_counter, (batch_audio, batch_labels) in enumerate(tepoch):
            audio, target_label = batch_audio.to(device), batch_labels.to(device)
            predictions = model(audio)

            loss = loss_function(predictions, target_label)
            train_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            correct_predictions += (predictions.argmax(1) == target_label).type(torch.float).sum().item()
            partial_metrics = {
                'loss': train_loss.item()/(batch_counter + 1),
                'accuracy': correct_predictions/((batch_counter + 1) * batch_size)
            }
            tepoch.set_postfix(partial_metrics)
        train_loss = train_loss.item() / num_batches
        train_accuracy = correct_predictions / size
    
    return train_loss, train_accuracy

# Validation/test loop (contains foward pass only, weights are not updated)
def val_loop(dataloader, model, loss_function, device):
    model.eval()

    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    val_loss, correct_predictions = 0, 0

    for batch_audio, batch_labels in dataloader:
        audio, target_label = batch_audio.to(device), batch_labels.to(device)
        predictions = model(audio)

        loss = loss_function(predictions, target_label)
        val_loss += loss

        correct_predictions += (predictions.argmax(1) == target_label).type(torch.float).sum().item()
    
    val_loss = val_loss.item() / num_batches
    val_accuracy = correct_predictions / size

    return val_loss, val_accuracy