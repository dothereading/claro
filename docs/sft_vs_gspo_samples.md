# SFT vs GSPO — random sample of 5

Side-by-side outputs on the same source for the SFT baseline (`sft_n1500_4b`) and the final GSPO model (`grpo_v11_4b_n1500`, checkpoint 200). Items are a fixed random sample (seed 42) of the 30-item held-out eval set — not cherry-picked. The parenthetical level is the DeepSeek mode-of-3 label on the SFT output.

Source files: `eval_results/sft_n1500_4b_eval30.json`, `eval_results/ckpt_n1500_0200.json`.

## 1. Melicope (SFT level: A2)

**Source:**

> Melicope is a genus of about 240 species of shrubs and trees in the family Rutaceae, occurring from the Hawaiian Islands across the Pacific Ocean to tropical Asia, Australia and New Zealand. Plants in the genus Melicope have simple or trifoliate leaves arranged in opposite pairs, flowers arranged in panicles, with four sepals, four petals and four or eight stamens and fruit composed of up to four follicles.

**SFT:**

> Melicope is a group of about 240 plants. These plants are shrubs and trees. They belong to the Rutaceae plant family. You can find these plants from the Hawaiian Islands to Asia, Australia, and New Zealand.

Melicope plants have simple or three-leaf pairs. The leaves grow in pairs on the stem. The plants have flowers that grow in long, thin branches called panicles. Each flower has three parts: four sepals, four petals, and four or eight sperm-carrying parts called stamens. The fruit is made of up to four small pods called follicles.

**GSPO (ckpt 200):**

> Melicope is a group of about 240 plants. They are shrubs and trees. They belong to the Rutaceae family. You can find them in Hawaii and other islands in the Pacific Ocean. They also grow in Asia, Australia, and New Zealand.

Melicope plants have simple or three-leaf leaves. The leaves grow in pairs on opposite sides of the stem. The flowers grow in long, thin groups called panicles. Each flower has four petals, four sepals, and four or eight pollen-carrying parts. The fruit has up to four small pods.

## 2. Selling Kabul (SFT level: A2)

**Source:**

> Selling Kabul is a 2019 American play by Sylvia Khoury. Set in 2013, the play centers a former interpreter for the U.S. Military during the War in Afghanistan forced to choose between witnessing the birth of his child or staying in hiding as a target of the Taliban. The play was a finalist for the 2022 Pulitzer Prize for Drama.

**SFT:**

> Selling Kabul is a play from 2019. Sylvia Khoury wrote it. The play is set in 2013. It is about a man who used to work for the U.S. Military in Afghanistan. He worked as an interpreter.

In 2013, the U.S. was at war in Afghanistan. The Taliban want to hurt him. He has to choose between his child and his own safety. He has to stay hidden.

In 2022, the play was a finalist for the Pulitzer Prize for Drama. This is a big award for plays.

**GSPO (ckpt 200):**

> Selling Kabul is a play from 2019. Sylvia Khoury wrote it. The play is set in 2013. It is about a man who used to work for the U.S. Military in Afghanistan. He is now a former interpreter. The Taliban want to hurt him. He must choose between seeing his child born and hiding. In 2022, the play was a finalist for the Pulitzer Prize for Drama.

## 3. Posad (SFT level: A2)

**Source:**

> A posad was a type of settlement in East Slavic lands between the 9th to 15th centuries, it was often surrounded by ramparts and a moat, adjoining a town or a kremlin, but outside of it, or adjoining a monastery. The term is derived from the Slavic prefix po- ('little') and sadъ, 'planted area, garden.' The posad was inhabited by craftsmen and merchants and was its own distinct community, separate from the city it adjoined. Some posads developed into towns, such as Pavlovsky Posad and Sergiev Posad.

**SFT:**

> A posad was a small settlement in East Slavic lands from the 9th to 15th centuries. It was often surrounded by walls and a moat. A town or a monastery was next to it, or next to the posad.

The word "posad" comes from two Slavic words. The first is "po-" which means "little". The second word is "sadъ", which means "a small area or garden".

The posad was home to craftsmen and merchants. They lived in their own community. This community was separate from the town it was next to. Some posads later became towns, like Pavlovsky Posad and Sergiev Posad.

**GSPO (ckpt 200):**

> A posad was a small settlement in East Slavic lands from the 9th to 15th centuries. It was often surrounded by walls and a ditch. It was next to a town or a kremlin, or next to a monastery.

The word "posad" comes from the Slavic words "po-" (little) and "sadъ" (planted area).

People who worked with tools and traders lived in the posad. They had their own community, separate from the town next to them.

Some posads grew into towns, like Pavlovsky Posad and Sergiev Posad.

## 4. Haemulon (SFT level: A2)

**Source:**

> Haemulon is a genus of fish in the grunt family known as the scaled-fin grunts. Most are found in the western Atlantic Ocean, with a few species known from the eastern Pacific Ocean. This genus is considered to be one of the most important fish groups of the coral reefs of Brazil due to its commercial value and crucial ecological role.

**SFT:**

> Haemulon is a group of fish called scaled-fin grunts. They belong to the grunt family. Most of these fish live in the western Atlantic Ocean. A few species also live in the eastern Pacific Ocean.

This group of fish is very important for coral reefs in Brazil. People catch them for food, and they also help the coral reefs stay healthy.

**GSPO (ckpt 200):**

> Haemulon is a group of fish. They belong to the grunt family. These fish have scales on their fins. Most of them live in the western Atlantic Ocean. A few live in the eastern Pacific Ocean.

This group of fish is very important in Brazil. They live on coral reefs. People catch them for food. They also help the coral reefs stay healthy.

## 5. Priyanka Karki (SFT level: A2)

**Source:**

> Priyanka Karki is a Nepalese actress, director and producer. The winner of the Miss Teen Nepal pageant, Karki is Nepal's highest-paid actress and has received numerous accolades, including two National Film Awards and five Kamana Film Awards. In 2023, the Government of Nepal honoured her with the Chamber of Commerce for her contribution to the Nepali film industry. Described in the media as one of the most successful actresses of Nepali cinema.

**SFT:**

> Priyanka Karki is an actress, director, and producer from Nepal. She won Miss Teen Nepal. She is also Nepal's highest-paid actress. She has won many awards, like two National Film Awards and five Kamana Film Awards. In 2023, the Government of Nepal gave her a special award for her work in the Nepali film industry. The media calls her one of the most successful actresses in Nepali cinema.

**GSPO (ckpt 200):**

> Priyanka Karki is an actress, director, and producer from Nepal. She won Miss Teen Nepal. She is also Nepal's highest-paid actress. She has won two National Film Awards and five Kamana Film Awards. In 2023, the government of Nepal gave her an award for her work in the Nepali film industry. The media calls her one of the most successful actresses in Nepali cinema.
