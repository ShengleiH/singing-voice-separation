# singing-voice-separation

I will try to use several different models to do singing voice separtion.

In 'singing voice separtion' task, the input to the model is mixed signal, the output from the model are two signals, one for vocal and the other for backgound music.

In the very early stage, now, I have just implemented the basic DNN model to do this task.

This DNN model is trained on ['MIR-1K' dataset][1] collected by Chao-Ling Hsu and Prof. Jyh-Shing Roger Jang from National Taiwan University. And my final results are approximated to:

GNSDR_music=5.7659940276222175

GSIR_music=11.613638774291942

GSAR_music=8.051359491045204

GNSDR_vocal=5.737649750251389

GSIR_vocal=13.23238571027173

GSAR_vocal=7.455369915779597

And I am trying to use GAN to do this task now.

# Reference:

1. Singing Voice Separation and Pitch Extraction from Monaural Polyphonic Audio Music Via DNN and Adaptive Pitch Tracking

2. SVSGAN: SINGING VOICE SEPARATION VIA GENERATIVE ADVERSARIAL NETWORK

[1]: https://sites.google.com/site/unvoicedsoundseparation/mir-1k
