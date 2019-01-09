# singing-voice-separation

I will try to use several different models to do singing voice separtion.

In 'singing voice separtion' task, the input to the model is mixed signal, the output from the model are two signals, one for vocal and the other for backgound music.

In the very early stage, now, I have just implemented the basic DNN model to do this task.

This DNN model is trained on ['MIR-1K' dataset][1] collected by Chao-Ling Hsu and Prof. Jyh-Shing Roger Jang from National Taiwan University. And my final results are approximated to:

| Metrics       | Music         | Vocal          |
|:-------------|:-------------| :-------------|
| GNSDR         | 5.766         | 5.738          |
| GSIR          | 11.614        | 13.232         |
| GSAR          | 8.051         | 7.455          |

If you have ideas to improve this results, please let me know. The results in paper \[1\] is better than this one, but I have no idea how to improve it.

And I am trying to use GAN to do this task now. (Well, unfortunately, it failed :(, the model cannot converge.)

# Reference:

1. Singing Voice Separation and Pitch Extraction from Monaural Polyphonic Audio Music Via DNN and Adaptive Pitch Tracking

2. SVSGAN: SINGING VOICE SEPARATION VIA GENERATIVE ADVERSARIAL NETWORK

[1]: https://sites.google.com/site/unvoicedsoundseparation/mir-1k
