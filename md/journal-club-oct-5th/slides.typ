// Compile from the repository root so the shared template is within the Typst root:
// typst compile --root . md/journal-club-oct-5th/slides.typ md/journal-club-oct-5th/slides.pdf

#import "../../writings/templates/slides/slide.typ": slide

#set document(title: "Journal club — 5 October 2026")
#slide(body: [
  #align(center + horizon)[
    #image("images/title.png", width: 100%, height: 148mm, fit: "contain")
  ]
])

// Talk outline: 45 minutes; questions afterwards.
// Hierarchy 8 min, Territory 8 min, Fighting 12 min, Alliances 9 min,
// Interpretations 8 min. Interpretation subsections remain undecided.
// 1. Hierarchy
// Source: Yan & Lin (2026), Introduction, Box 1, residency and fighting
// sections, and Figure 1, pp. 495–498.
#slide(header: "Hierarch - What, why and how it varies", body: [
  + *Rank and relationships* — The highest-ranking chicken pecks all others;
    the second-ranking bird pecks everyone except the highest.
  + *Why hierarchies form* — Once mice establish dominance relationships,
    overt aggression declines.
  + *Different forms* — Female mouse hierarchies are generally less linear
    and less despotic than male hierarchies.
  + *Deep evolutionary roots* — Hierarchies occur in crickets, cichlids,
    chickens, mice and chimpanzees.
  + *Unequal benefits and costs* — Dominant male cichlids gain reproductive
    opportunities but have lower feeding and growth rates; subordinates are
    reproductively suppressed.
])
#slide(header: "Hierarch - Three example routes to status", body: [
  #align(center + horizon)[
    #image("images/yan-lin-2026-fig-04.png", width: 100%, height: 128mm, fit: "contain")
  ]
])
// Source: Yan & Lin (2026), mechanistic sections and Figure 4.
#slide(header: "Hierarch - The brain’s role in hierarchy", body: [
  #set enum(spacing: 12pt)
  + *Territory*
    + *Social behaviour network (SBN):* basic social behaviour circuits.
    + *Hypothalamic–pituitary–gonadal (HPG) axis:* reproductive hormone pathway.
  + *Fighting*
    + *Posterior ventrolateral ventromedial hypothalamus (pVMHvl):* aggression.
    + *Anterior VMHvl (aVMHvl):* defence and avoidance.
    + *Caudal medial preoptic area (cMPOA):* suppression of aggression.
    + *Ventral tegmental area (VTA), nucleus accumbens (NAc), basolateral amygdala (BLA):*
      reward and threat learning.
  + *Social alliances*
    + *Medial prefrontal cortex (mPFC), anterior cingulate cortex (ACC):*
      social assessment and learning.
    + *Temporoparietal junction (TPJ), posterior superior temporal sulcus (pSTS):*
      human mental-state inference and social cues.
])

// 2. Territory
// Source: Yan & Lin (2026), Residency-based social hierarchy, p. 497.
#slide(header: "Territory - Ownership gives residents an advantage", body: [
  + *Resident advantage* — In several bird species, residents tend to defeat
    intruders; body size alone does not determine the outcome.
  + *Territory enables dominance* — In African cichlid fish, territory holders
    defend spawning sites where females lay eggs, court females and reproduce.
  + *Ownership can change status* — Removing a dominant cichlid allows a subordinate
    to occupy its territory and begin defending it within minutes.
  + *Fighting still contributes* — Residents use chasing and threat displays;
    a vacant territory can trigger competition among several males.
])
// Focus on Figure 1a (appearance and behaviour).
#slide(header: "Territory - A territorial opportunity changes behaviour", body: [
  #align(center + horizon)[
    #image("images/yan-lin-2026-fig-01.png", width: 100%, height: 128mm, fit: "contain")
  ]
])
// Source: Yan & Lin (2026), Residency-based social hierarchy, p. 497.
#slide(header: "Territory - Hormones adapt to changing status", body: [
  + *Reproductive hormone pathway* — Status ascent activates the
    hypothalamic–pituitary–gonadal (HPG) axis, linking the brain to reproductive
    hormone production.
  + *Rapid ascent response* — Within 30 minutes, circulating sex steroids increase
    and steroid receptors are upregulated in hypothalamic regions involved in
    aggression and reproduction.
  + *Stable status differences* — Dominant male African cichlid fish maintain
    higher testosterone; subordinates have suppressed testosterone, elevated
    cortisol and reduced reproductive capacity.
  + *Response to displacement* — After a dominant male loses its territory,
    subordinate behaviour appears within approximately 30 minutes; cortisol
    rises within 24 hours.
])
// Source: Yan & Lin (2026), Residency-based social hierarchy and Figure 1b,
// pp. 497–498. Gene expression does not establish the initiating mechanism.
#slide(header: "Territory - The social behaviour network responds", body: [
  + *Distributed response* — Status transitions alter gene expression across the
    social behaviour network, including preoptic and hypothalamic regions.
  + *Ascending males* — The immediate early genes _cfos_ and _egr-1_ increase
    together across multiple network regions.
  + *Descending males* — Gene-expression changes follow a different, more selective
    pattern, including the preoptic area (POA) and anterior tuberal nucleus (ATn).
  + *Distinct transitions* — Gaining and losing status recruit different expression
    patterns; descent is not simply the reverse of ascent.

  #v(18pt)
  #text(weight: "bold")[A reinforcing loop]
  #v(10pt)
  Territorial opportunity → neural response → dominant behaviour → social feedback
  → further neural adaptation
  #v(10pt)
  Hormonal changes ↔ neural circuits and behaviour
  #v(12pt)
  #text(size: 12pt)[
    The review supports hormonal reinforcement and experience-dependent adaptation;
    the complete feedback loop is a synthesis, and the initiating mechanism remains unclear.
  ]
])
// Diagram and terminology for the reinforcing-loop synthesis.
#slide(header: "Territory - Neural and hormonal reinforcement", body: [
  #grid(
    columns: (1.5fr, 1fr),
    gutter: 8mm,
    align: horizon,
    image("images/territory-status-loop.svg", width: 100%, height: 128mm, fit: "contain"),
    [
      #set text(size: 13pt)
      #set enum(spacing: 14pt)
      + *Social behaviour network (SBN)* — Brain regions coordinating basic social
        behaviours, including aggression and reproduction.
      + *Preoptic and hypothalamic regions* — Brain areas linking social behaviour
        with bodily and hormonal responses.
      + *cfos and egr-1* — Genes whose increased expression marks a neural response.
      + *Hypothalamic–pituitary–gonadal (HPG) axis* — Hormone pathway linking the
        brain, pituitary gland and reproductive glands.
      + *Sex steroids* — Hormones, including testosterone, that influence
        reproductive function and behaviour-related circuits.
      + *Social feedback* — Consequences of interactions that may influence later
        behaviour and neural adaptation; the dashed loop is proposed.
    ],
  )
])
// Source: Yan & Lin (2026), Residency-based social hierarchy, p. 497.
#slide(header: "Territory - How does opportunity become a status transition?", body: [
  + *Social opportunity* — Removing a dominant male makes its territory available;
    one or several subordinate males may respond.
  + *Rapid transition* — Within minutes, an ascending male can adopt dominant
    colouration, occupy the territory and defend it.
  + *Coordinated adaptation* — The review describes neural initiation followed by
    hormonal reinforcement, alongside changes in social behaviour network gene expression.
  + *Unresolved mechanism* — How the brain detects territorial opportunity and
    initiates these changes remains poorly understood.
])

// 3. Fighting
// Source: Yan & Lin (2026), Fighting-based social hierarchy, pp. 498–499.
#slide(header: "Fighting - Fighting outcomes shape future encounters", body: [
  + *Winner effect* — Winning increases readiness to attack; after repeated
    victories, this can extend to other opponents.
  + *Loser effect* — Defeat reduces willingness to escalate subsequent aggressive
    encounters.
  + *Opponent-specific learning* — Defeated animals associate an opponent’s cues
    with the aversive experience and avoid that individual.
  + *Hierarchy emerges over encounters* — Repeated interactions establish
    dominance relationships as animals adjust whom they challenge and whom they yield to.
])
// Focus on the winner portion of Figure 2.
#slide(header: "Fighting - Winning strengthens aggression circuits", body: [
  #align(center + horizon)[
    #image("images/yan-lin-2026-fig-02.png", width: 100%, height: 128mm, fit: "contain")
  ]
])
// Source: Yan & Lin (2026), Role of social behaviour network and Figure 2,
// pp. 499–500. These circuit findings concern mice.
#slide(header: "Fighting - Losing suppresses attacks and teaches avoidance", body: [
  + *Suppressing aggression* — After defeat, aggressor cues recruit
    *caudal medial preoptic area (cMPOA)* neurons that inhibit aggression-driving
    *posterior VMHvl (pVMHvl)* neurons.
  + *Learning whom to avoid* — Pain during defeat triggers oxytocin release,
    strengthening aggressor-cue inputs onto oxytocin-receptor neurons in the
    *anterior VMHvl (aVMHvl)*.
  + *Opponent-specific memory* — A single 10-minute defeat can reduce investigation
    of, and proximity to, the aggressor the following day.
  + *Causal evidence* — Inhibiting the cMPOA neurons increases attacks on stronger
    opponents; activating the aVMHvl neurons induces avoidance even in undefeated mice.

  #v(12pt)
  #text(size: 12pt)[
    VMHvl: ventrolateral part of the ventromedial hypothalamus. These findings concern mouse circuits.
  ]
])
#slide(header: "Fighting - Defeat recruits broader learning circuits", body: [
  #align(center + horizon)[
    #image("images/yan-lin-2026-fig-03.png", width: 100%, height: 128mm, fit: "contain")
  ]
])
// Source: Yan & Lin (2026), Role of mesolimbic circuit, pp. 500–502.
// Contextual-memory mechanisms partly draw on broader fear-learning evidence.
#slide(header: "Fighting - Defeat recruits broader learning circuits", body: [
  + *Teaching signals* — Dopamine from the *ventral tegmental area (VTA)* helps
    reshape responses to aggressor cues and defeat-associated contexts.
  + *Opponent and place memories* — The *basolateral amygdala (BLA)* supports
    learning about threatening social cues; the *ventral hippocampus (vHip)*
    contributes contextual information.
  + *Reduced approach* — Dopamine decreases in the *nucleus accumbens (NAc)*
    during attacks, favouring plasticity that reduces subsequent approach.
  + *Increased avoidance* — Dopamine in the *tail of the striatum (TS)* increases
    when defeated mice approach an aggressor, supporting threat avoidance.
])

// 4. Alliances
// Source: Yan & Lin (2026), Routes to high social status and Alliance-based
// social hierarchy, pp. 495–496 and 503. Alliances can supplement fighting.
#slide(header: "Alliances - Social support can outweigh individual strength", body: [
  + *Coalition support* — Partners help individuals in conflicts and deter
    challengers, allowing physically modest individuals to attain high rank.
  + *Chimpanzees* — Physical attributes can influence initial rank, while
    enduring dominance is supported by alliances.
  + *Bottlenose dolphins* — Strong, stable alliances among largely unrelated
    males improve access to females and reproductive success.
  + *Relationships change rank* — Across several species, rank changes can
    follow shifts in coalition structure rather than changes in physical condition.
])
// Source: Yan & Lin (2026), Social intelligence across species, pp. 503–504.
#slide(header: "Alliances - Building alliances requires social intelligence", body: [
  + *Assess others* — Recognize individuals and track their behaviour,
    capabilities and emotional states.
  + *Learn by observation* — Rhesus monkeys can learn dominance relationships
    by watching interactions without participating.
  + *Adjust cooperation* — In cooperative tasks, dominant marmosets sometimes
    follow subordinate partners; leading and following need not match rank.
  + *Infer mental states* — Humans use theory of mind to anticipate others’
    beliefs, intentions and responses; comparable abilities in other species remain debated.
])
// Functional synthesis; social-system links are interpretations, not neural connections.
#slide(header: "Alliances - Human neural networks and social systems", body: [
  #grid(
    columns: (1.5fr, 1fr),
    gutter: 8mm,
    align: horizon,
    image("images/human-social-circuits.svg", width: 100%, height: 128mm, fit: "contain"),
    [
      #set text(size: 12pt)
      #set enum(spacing: 12pt)
      + *Posterior superior temporal sulcus (pSTS)* — Processes movement and
        changing facial expressions.
      + *Temporoparietal junction (TPJ)* — Helps infer others’ beliefs and intentions.
      + *Medial prefrontal cortex (mPFC)* — Supports social assessment and reasoning
        about others.
      + *Dorsomedial prefrontal cortex (dmPFC)* — Part of mPFC involved in
        mental-state inference.
      + *Anterior cingulate cortex / gyrus (ACC / ACCg)* — Represents others’
        distress and rewards; supports social learning. Evidence spans species.
      + *Norms and institutions* — Shared expectations and rules that organize
        support and authority; social systems, rather than brain regions.
    ],
  )
])
// Source: Yan & Lin (2026), Table 1. Clip the image at display time to hide
// its title and abbreviation footnote; preserve the original image unchanged.
#slide(header: "Alliances - Neural substrates of social intelligence", body: [
  #grid(
    columns: (1fr, 1.5fr),
    gutter: 8mm,
    align: horizon,
    align(center)[
      #box(width: 90mm, height: 106.5mm, clip: true)[
        #place(top + left, dy: -10mm)[
          #image("images/yan-lin-2026-table-01.png", width: 90mm, height: 130.1mm, fit: "contain")
        ]
      ]
    ],
    [
      #set text(size: 11.5pt)
      #set enum(spacing: 7pt)
      + *Prelimbic cortex (PL)* — Rodent frontal region involved in monitoring others.
      + *Anterior cingulate cortex (ACC)* — Supports social assessment, emotion
        processing and observational learning.
      + *Medial prefrontal cortex (mPFC)* — Frontal network representing information
        about oneself and others.
      + *Dorsomedial prefrontal cortex (dmPFC)* — Upper medial frontal region involved
        in social evaluation and mental-state inference.
      + *Ventromedial prefrontal cortex (vmPFC)* — Lower medial frontal region
        contributing to valuation and mental-state inference.
      + *Ventral tegmental area (VTA)* — Source of dopamine signals involved in learning.
      + *Nucleus accumbens (NAc)* — Region involved in reward and approach behaviour.
      + *Basolateral amygdala (BLA)* — Region linking cues to emotionally important outcomes.
      + *Posterior superior temporal sulcus (pSTS)* — Processes movement and facial cues.
      + *Temporoparietal junction (TPJ)* — Helps infer others’ beliefs and intentions.
      + *Theory of mind (ToM)* — Inferring another person’s mental state; an ability,
        rather than a brain region.
    ],
  )
])
// Source: Yan & Lin (2026), Different routes to high social status in humans,
// p. 497. Conceptual synthesis of social processes, not an established neural circuit.
#slide(header: "Alliances - Human status involves prestige, alliances and institutions", body: [
  + *Dominance and prestige* — Influence can arise through coercion or through
    respect voluntarily given for competence and achievement.
  + *Alliances amplify prestige* — The authors argue that expertise alone does
    not guarantee high status; relationship networks help sustain recognition and support.
  + *Institutions formalize authority* — Collective endorsement can turn prestige
    and social support into roles such as executive or director.
  + *Authority reinforces status* — Formal positions increase visibility and
    access to networks, helping sustain further influence.

  #v(18pt)
  *Authors’ proposed cycle:* prestige → alliances → institutional status
  → expanded visibility and networks.
])

// 5. Interpretations: text only; subsections to be decided.
#slide(header: "Interpretations - Interpretations", body: [
  The review’s human neural evidence mainly explains capacities for navigating
  social relationships, while its account of institutions leaves room for our
  interpretation that collective action can organize those capacities towards
  different distributions of status and authority.
])
