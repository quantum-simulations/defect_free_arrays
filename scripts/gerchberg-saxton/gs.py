# import math
import numpy as np
import time

def Ger_Sax_algo(img, max_iter):
    """    Gerchberg-Saxton algorithm for phase retrieval.
    Args:
        img (numpy.ndarray): Input image (grayscale).
        max_iter (int): Number of iterations for the algorithm.
    Returns:
        numpy.ndarray: Phase mask obtained from the algorithm.
    """
    total_start_time = time.time()

    h, w = img.shape

    # initialize phase and amplitude masks in both spatial and fourier domains
    pm_s = np.random.rand(h, w)
    pm_f = np.ones((h, w))
    am_s = np.sqrt(img)
    am_f = np.ones((h, w))

    # the signal in the spatial domain takes initially the image amplitude and the random phase mask
    signal_s = am_s*np.exp(pm_s * 1j)

    # lists to store times
    iter_times = []
    fft_times = []
    ifft_times = []

    for iter in range(max_iter):
        iter_start = time.time()
        fft_start = time.time()

        # fft of the signal in the spatial domain --> signal in the fourier domain
        signal_f = np.fft.fft2(signal_s)

        fft_end = time.time()
        fft_times.append(fft_end - fft_start)

        # we put the fourier signal amplitude to 1 and keep the phase
        pm_f = np.angle(signal_f)
        signal_f = am_f*np.exp(pm_f * 1j)

        ifft_start = time.time()

        # ifft of the signal in the fourier domain --> signal in the spatial domain
        signal_s = np.fft.ifft2(signal_f)
        ifft_end = time.time()
        ifft_times.append(ifft_end - ifft_start)

        pm_s = np.angle(signal_s)
        signal_s = am_s*np.exp(pm_s * 1j)

        iter_end = time.time()
        iter_times.append(iter_end - iter_start)

    pm = pm_f

    total_end_time = time.time()
    print(f"Ger_Sax_algo executed in {total_end_time - total_start_time:.6f} s")
    print(f"Average iteration time: {1000 * np.mean(iter_times):.6f} ms")
    print(f"Average FFT time: {1000 * np.mean(fft_times):.6f} ms")
    print(f"Average IFFT time: {1000 * np.mean(ifft_times):.6f} ms")

    return pm

