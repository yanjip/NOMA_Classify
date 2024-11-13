# time: 2024/11/8 11:26
# author: YanJP
import numpy as np
def channel():
    def path_loss(d):
        return 128.1 + 37.6 * np.log10(d)
    # Set the number of channels
    num_channels = 10
    # Simulate distances (assuming random distances for this example)
    distances = np.random.uniform(0.1, 0.5, num_channels )   # Distances in kilometers
    # Calculate path loss for each channel
    path_loss_dB = path_loss(distances)
    # Apply log-normal shadowing (10 dB standard deviation)
    shadowing = np.random.normal(scale=5, size=num_channels)
    shadowing=np.clip(shadowing,-5,5)
    path_loss_linear = np.power(10, -path_loss_dB / 10)
    lar_fad=np.sqrt(path_loss_linear) *np.exp(-shadowing / 10)
    return  lar_fad

if __name__ == '__main__':
    lar_fad=channel()