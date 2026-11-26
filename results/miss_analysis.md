# Why the best system misses validation Recall@1

System: bge-small look-alike first stage + MiniLM-L12 cross-encoder reranking the top 20. Validation Llama questions: 500, Recall@1 0.732, 134 misses.

Blind pairwise check: Llama 3.1 8B (4-bit, MLX) sees the question and two entries (title | url), the answer and the system's top 1, without being told which is correct, and answers A, B or CANT_TELL. It is asked twice with A and B swapped (first order set by seeded hash, `answer_first` in the csv). `knowledge`: picks the answer both times; `wrong`: picks the top 1 both times; `ambiguous`: CANT_TELL or a different page in the two orders.

| Class | Misses | Share of misses | Share of questions |
|---|---|---|---|
| knowledge | 23 | 17% | 4.6% |
| ambiguous | 109 | 81% | 21.8% |
| wrong | 2 | 1% | 0.4% |

Implied Recall@1 ceiling for title + url: 0.732 + 23/500 = **0.778** (only `knowledge` misses count as fixable; this is optimistic, since one fixable pair can still lose to rank 2 or 3).

`duplicate` (rule, not the LLM: same title on the same site, or the same URL without the query string): 5 misses, by class: knowledge 2, ambiguous 2, wrong 1. They are candidates for an evaluation fix. The rule also flags different pages that share a generic title ("Overview" on mayoclinic.org) or keep their id in the query string (play.google.com, amazon nodeId).

Position bias: over both asks Llama answered A 89 times, B 28 times and CANT_TELL 151 times, so a consistent pick is rare and the Llama ceiling is a lower bound.

Hand check of 30 misses (sampled with seed 1, labelled blind in the first-ask order before seeing Llama's picks): Llama agrees on 53%. With the hand labels' knowledge share (43%) the ceiling is 0.848.

Rows: hand label, columns: Llama label.

```
miss_class  ambiguous
hand_class           
ambiguous          16
knowledge          13
wrong               1
```

## knowledge: examples

- Q: What's the name of that bird I read about that runs fast?  
  answer (rank 4): `Greater roadrunner | en.wikipedia.org/wiki/Greater_Roadrunner`  
  top 1: `Greater trochanter | en.wikipedia.org/wiki/Greater_trochanter`
- Q: What's the basic shape for making an origami bird that I saw on that origami website?  
  answer (rank 3): `Mountain and Valley Folds | origami.me/beginners-guide`  
  top 1: `How to make origami flowers | origami-flower.org/howto-origami-flower.php`
- Q: How do I update the software on my Kindle?  
  answer (rank 2): `Fire & Kindle Software Updates | amazon.com/gp/help/customer/display.html?nodeId=200529680`  
  top 1: `Kindle Fire (1st Generation) Software Updates | amazon.com/gp/help/customer/display.html?nodeId=200790620` (duplicate)
- Q: What's the address of the CNN building I saw in Atlanta?  
  answer (rank 2): `CNN Center | en.wikipedia.org/wiki/CNN_Center`  
  top 1: `Experience Atlanta | omnihotels.com/hotels/atlanta-cnn-center`
- Q: What's the Wikipedia article about the wife of President Kennedy that I looked at?  
  answer (rank 2): `Jacqueline Kennedy Onassis | en.wikipedia.org/wiki/Jacqueline_Kennedy_Onassis`  
  top 1: `Ethel Kennedy | en.wikipedia.org/wiki/Ethel_Kennedy`
- Q: What are those old plastic figurines I saw on Wikipedia called?  
  answer (rank 2): `Troll doll | en.wikipedia.org/wiki/Troll_doll`  
  top 1: `Goo Goo Dolls | en.wikipedia.org/wiki/Goo_Goo_Dolls`
- Q: What's the name of that UFC reality TV show I watched?  
  answer (rank 2): `The Ultimate Fighter | en.wikipedia.org/wiki/The_Ultimate_Fighter`  
  top 1: `Ultimate Fighting Championship | en.wikipedia.org/wiki/Ultimate_Fighting_Championship`
- Q: What's the name of that TV show about a powerful crime family?  
  answer (rank 3): `McMafia | en.wikipedia.org/wiki/McMafia`  
  top 1: `Lederhosen | en.wikipedia.org/wiki/Lederhosen`
- Q: What's the information I read on building bat boxes?  
  answer (rank 3): `Bat House | eparks.org/wildlife_protection/wildlife_facts/bats/bat_house.asp`  
  top 1: `Batting Cages | battingcagesinc.com`
- Q: What are some of the mythical creatures I saw listed on that Wikipedia page?  
  answer (rank 2): `List of Greek mythological creatures | en.wikipedia.org/wiki/List_of_Greek_mythological_creatures`  
  top 1: `List of mythological objects | en.wikipedia.org/wiki/List_of_mythological_objects`

## ambiguous: examples

- Q: How do I use arrays within arrays in JavaScript?  
  answer (rank 2): `JavaScript Arrays | w3schools.com/js/js_arrays.asp`  
  top 1: `Array | developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array`
- Q: Where is the blank OPM form 71 that I was looking at?  
  answer (rank 2): `Forms OPM Forms | opm.gov/forms/opm-forms`  
  top 1: `Forms | opm.gov/Forms`
- Q: What did the critics say about the John Wick movie I looked at?  
  answer (rank 5): `John Wick Photos | rottentomatoes.com/m/john_wick`  
  top 1: `John Wick | en.wikipedia.org/wiki/John_Wick_(film)`
- Q: What are the benefits of taking a boron supplement that article mentioned?  
  answer (rank 2): `Boron Uses â Boost Bone Density and Much More | draxe.com/boron-uses`  
  top 1: `Boron | examine.com/supplements/boron`
- Q: What's the postal code for Guelph?  
  answer (rank 7): `Find a Postal Code | canadapost.ca/cpo/mc/personal/postalcode/fpc.jsf`  
  top 1: `Postal code | en.wikipedia.org/wiki/Postal_code`
- Q: What does the word respect mean, like the dictionary page I looked at?  
  answer (rank 2): `respect | merriam-webster.com/dictionary/respect`  
  top 1: `respect | dictionary.com/browse/respect`
- Q: What are the new ID requirements for flying that I read about last year?  
  answer (rank 3): `Identification | tsa.gov/travel/security-screening/identification`  
  top 1: `Identification Requirements for Domestic Air Travel in the United States | traveltips.usatoday.com/identification-requirements-domestic-air-travel-united-states-64080.html`
- Q: What's the current price per pound of scrap aluminum cans?  
  answer (rank 2): `Scrap Metal Prices Today USA - Check Today's Scrap Metal Prices Per Pound | scrapmetalpricestodayusa.com`  
  top 1: `Today's Current Price is $.40 per pound | canpacrecycling.com/pricing.html`
- Q: What are those special boxes called that keep chemicals safe?  
  answer (rank 6): `A complete selection of products designed to meet the worldâs highest standards | hazmatpac.com`  
  top 1: `Our mission Weâre here to unleash the worldâs creative energy by designing a more enlightened way of working. | dropbox.com/about`
- Q: What's the salary of that soccer player I was reading about on his website?  
  answer (rank 3): `Neymar salary | neymarjr.net/salary`  
  top 1: `Cristiano Ronaldo Salary | ronaldo7.net/salary/cristiano-ronaldo-salary.html`

## wrong: examples

- Q: What's the article I read about taking a gap year supposed to be about?  
  answer (rank 2): `What is a Gap Year? | topuniversities.com/blog/what-gap-year`  
  top 1: `Should You Take A Gap Year? | forbes.com/sites/francesbridges/2014/04/29/should-you-take-a-gap-year`
- Q: What's the current snow report from Whistler?  
  answer (rank 2): `Snow and Weather Report | whistlerblackcomb.com/the-mountain/mountain-conditions/snow-and-weather-report.aspx`  
  top 1: `Snow and Weather Report | whistlerblackcomb.com/mountain-info/snow-report` (duplicate)
