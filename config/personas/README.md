# Personas

A from-scratch model can't follow written instructions the way a chatbot
does, so each persona is defined by two things the model *can* do:

- **`lead_in`**: the persona writes its position by continuing
  `"<claim> <lead_in>"`, in the model's own words.
- **`probes`**: short phrases the persona is scored on. Its **signal** is how
  much more likely the model finds those phrases right after the claim
  than after a neutral lead-in (pointwise mutual information, in
  nats per token). Positive means "the claim pulls toward these words".
  `counter_probes`, if present, are subtracted.

Edit the phrases to change what a persona listens for. Keep the leading
space; it matches how words are tokenized mid-sentence.
