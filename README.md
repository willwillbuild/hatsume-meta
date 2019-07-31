# Hatsume-meta
Support repo for hatsume for research on how best to find highlights in game videos

Below are some of the approaches being explored

## Edge Counting

Use a laplacian filter to get the edges in the image.  Simple and easy to implement, and provides some decent results.  

The approach is ultimately at the mercy of the UI; in-game detection of action produces flashier UI (such as when defeating an enemy in a game).  Thus more subjective, yet equally interesting, action sequences cannot be detected effectively (for example a humorous or "cool" event).

## Bi-directional LSTM

Use Inception v3 network to first extract features from frames, then feed to BiRNN structure.

Has the advantage that subjective events can be defined and trained on.  But requires massive amount of data and labeling.

## Pogchampnet

Individually analyzes frames for excitement value

Naive approach that can be straightforward to implement in webapp

Still requires much data, but if just using indices for how interesting an event is, may not require as much data as the BiRNN.

Borrowed from below URL

https://medium.com/@farzatv/pogchampnet-how-we-used-twitch-chat-deep-learning-to-create-automatic-game-highlights-with-only-61ed7f7b22d4
