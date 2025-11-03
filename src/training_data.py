"""
Labeled Training Data for the Semantic Front-End
================================================

This module contains a curated list of sentences, each manually labeled
with its corresponding PhiCoordinate (Love, Justice, Power, Wisdom).

Format: (sentence, (love, justice, power, wisdom))
Each coordinate ranges from 0.0 to 1.0

This expanded dataset (500+ examples) provides comprehensive coverage
of the 4D semantic space to prevent overfitting and improve generalization.
"""

TRAINING_DATA = [
    # ========================================================================
    # HIGH LOVE - Compassion, Kindness, Mercy, Care, Nurturing
    # ========================================================================

    # Pure Love expressions
    ("True love is compassionate and kind.", (0.9, 0.7, 0.5, 0.8)),
    ("She showed great compassion for the suffering.", (0.9, 0.6, 0.4, 0.7)),
    ("His kindness was a gift to everyone he met.", (0.9, 0.7, 0.3, 0.6)),
    ("Mercy triumphs over judgment.", (0.95, 0.6, 0.3, 0.7)),
    ("Her heart overflowed with unconditional love.", (0.95, 0.5, 0.2, 0.5)),
    ("He embraced them with warmth and acceptance.", (0.9, 0.6, 0.3, 0.6)),
    ("Love covers a multitude of wrongs.", (0.95, 0.5, 0.2, 0.7)),
    ("She cared for the orphans with tender devotion.",
     (0.95, 0.7, 0.4, 0.7)),
    ("His gentle spirit brought comfort to all.", (0.9, 0.6, 0.3, 0.7)),
    ("The mother's love knew no boundaries.", (0.95, 0.5, 0.4, 0.6)),

    # Love with wisdom
    ("Loving-kindness combined with understanding.", (0.85, 0.6, 0.3, 0.85)),
    ("Compassionate wisdom guides the heart.", (0.85, 0.6, 0.3, 0.9)),
    ("She loved them enough to teach them truth.", (0.8, 0.7, 0.4, 0.85)),
    ("Merciful correction comes from love.", (0.85, 0.65, 0.5, 0.8)),
    ("Wise love knows when to give and when to withhold.",
     (0.8, 0.6, 0.3, 0.9)),

    # Love with justice
    ("Love must be rooted in truth and righteousness.",
     (0.85, 0.85, 0.4, 0.75)),
    ("Righteous love confronts evil with grace.", (0.8, 0.8, 0.5, 0.7)),
    ("Her compassion was tempered by moral clarity.",
     (0.8, 0.85, 0.4, 0.75)),
    ("Just love does not tolerate abuse.", (0.75, 0.9, 0.5, 0.7)),
    ("Faithful love upholds the truth.", (0.85, 0.85, 0.4, 0.7)),

    # Nurturing and care
    ("She nurtured the children with patience.", (0.9, 0.6, 0.3, 0.7)),
    ("His care for the sick was selfless.", (0.95, 0.7, 0.3, 0.6)),
    ("They built a home filled with warmth.", (0.85, 0.6, 0.4, 0.6)),
    ("Love creates safe spaces for growth.", (0.9, 0.6, 0.4, 0.75)),
    ("Her presence brought healing to broken hearts.", (0.95, 0.6, 0.3, 0.7)),

    # ========================================================================
    # HIGH JUSTICE - Fairness, Righteousness, Integrity, Truth, Moral Order
    # ========================================================================

    # Pure Justice expressions
    ("A just society is built on fairness and truth.",
     (0.7, 0.9, 0.6, 0.8)),
    ("He acted with integrity in all his dealings.", (0.6, 0.9, 0.7, 0.8)),
    ("The judge delivered a fair and righteous verdict.",
     (0.5, 0.9, 0.8, 0.7)),
    ("Justice demands equal treatment under law.", (0.5, 0.95, 0.7, 0.75)),
    ("Righteousness exalts a nation.", (0.6, 0.95, 0.6, 0.8)),
    ("She stood firm for what is right.", (0.6, 0.9, 0.7, 0.75)),
    ("Truth and justice are inseparable.", (0.5, 0.95, 0.5, 0.85)),
    ("He fought corruption with unwavering integrity.",
     (0.6, 0.9, 0.6, 0.75)),
    ("Fair scales and honest weights.", (0.5, 0.95, 0.5, 0.7)),
    ("The righteous care about justice for the poor.",
     (0.75, 0.9, 0.4, 0.7)),

    # Justice with wisdom
    ("Wise judgment considers all perspectives.", (0.6, 0.85, 0.6, 0.9)),
    ("Discerning justice weighs evidence carefully.", (0.5, 0.9, 0.5, 0.85)),
    ("Thoughtful fairness seeks the truth.", (0.6, 0.85, 0.5, 0.9)),
    ("Understanding brings balanced judgment.", (0.6, 0.8, 0.5, 0.9)),
    ("She judged with wisdom and equity.", (0.6, 0.9, 0.6, 0.85)),

    # Justice with power
    ("Righteous authority enforces the law.", (0.5, 0.9, 0.85, 0.7)),
    ("Justice backed by strength protects the innocent.",
     (0.6, 0.9, 0.85, 0.7)),
    ("He wielded power to establish fairness.", (0.5, 0.9, 0.9, 0.6)),
    ("Strong enforcement of moral standards.", (0.5, 0.85, 0.9, 0.7)),
    ("Authority used to uphold righteousness.", (0.5, 0.9, 0.9, 0.65)),

    # Truth and honesty
    ("Speak the truth in all circumstances.", (0.6, 0.9, 0.4, 0.8)),
    ("Honesty is the foundation of trust.", (0.6, 0.9, 0.4, 0.75)),
    ("He never compromised on truthfulness.", (0.5, 0.95, 0.5, 0.8)),
    ("Integrity means doing right when unseen.", (0.6, 0.9, 0.5, 0.8)),
    ("Her word was her bond.", (0.6, 0.95, 0.5, 0.7)),

    # ========================================================================
    # HIGH POWER - Authority, Strength, Capability, Sovereignty, Command
    # ========================================================================

    # Pure Power expressions
    ("The king had absolute power over his domain.", (0.3, 0.5, 0.9, 0.6)),
    ("The storm was a powerful and unstoppable force.",
     (0.1, 0.3, 0.9, 0.2)),
    ("She spoke with authority and conviction.", (0.4, 0.6, 0.9, 0.7)),
    ("His strength was legendary and unmatched.", (0.3, 0.4, 0.95, 0.5)),
    ("They commanded vast armies with decisiveness.",
     (0.3, 0.5, 0.95, 0.6)),
    ("The empire's might extended across continents.",
     (0.2, 0.4, 0.95, 0.5)),
    ("She wielded extraordinary influence.", (0.4, 0.5, 0.9, 0.7)),
    ("His presence commanded respect and obedience.", (0.3, 0.6, 0.9, 0.6)),
    ("Raw power surged through the system.", (0.1, 0.3, 0.95, 0.3)),
    ("The earthquake demonstrated nature's awesome force.",
     (0.1, 0.2, 0.95, 0.2)),

    # Power with wisdom
    ("Wise leadership channels power effectively.", (0.4, 0.6, 0.85, 0.9)),
    ("Strength tempered by understanding.", (0.4, 0.6, 0.85, 0.85)),
    ("Intelligent use of authority.", (0.3, 0.6, 0.9, 0.9)),
    ("Capability guided by insight.", (0.4, 0.5, 0.85, 0.9)),
    ("She ruled with both might and discernment.", (0.4, 0.6, 0.9, 0.85)),

    # Power with justice
    ("Righteous power defends the weak.", (0.6, 0.85, 0.9, 0.7)),
    ("Just authority maintains order.", (0.5, 0.9, 0.9, 0.7)),
    ("He used his strength to protect the vulnerable.",
     (0.7, 0.8, 0.9, 0.6)),
    ("Fair enforcement of necessary discipline.", (0.5, 0.85, 0.9, 0.7)),
    ("Moral strength stands against corruption.", (0.6, 0.9, 0.85, 0.75)),

    # Dominance and control
    ("He dominated the competition completely.", (0.2, 0.4, 0.95, 0.5)),
    ("Absolute control over the situation.", (0.2, 0.5, 0.95, 0.6)),
    ("Her influence was overwhelming.", (0.3, 0.5, 0.9, 0.6)),
    ("Command presence filled the room.", (0.3, 0.5, 0.9, 0.6)),
    ("Decisive action with immediate effect.", (0.3, 0.5, 0.95, 0.7)),

    # ========================================================================
    # HIGH WISDOM - Knowledge, Understanding, Insight, Discernment, Teaching
    # ========================================================================

    # Pure Wisdom expressions
    ("The philosopher shared his profound wisdom.", (0.6, 0.7, 0.5, 0.9)),
    ("True wisdom is knowing you know nothing.", (0.7, 0.8, 0.4, 0.9)),
    ("She offered wise counsel to the young leader.",
     (0.7, 0.7, 0.6, 0.9)),
    ("Understanding comes from deep reflection.", (0.5, 0.6, 0.3, 0.95)),
    ("Knowledge and insight guide the path.", (0.5, 0.7, 0.4, 0.95)),
    ("He studied the ancient texts with discernment.",
     (0.5, 0.7, 0.4, 0.9)),
    ("Wisdom sees beyond surface appearances.", (0.6, 0.7, 0.3, 0.95)),
    ("Her teaching illuminated complex truths.", (0.6, 0.7, 0.5, 0.9)),
    ("The sage pondered life's deepest questions.", (0.6, 0.6, 0.3, 0.95)),
    ("Discernment distinguishes good from evil.", (0.6, 0.8, 0.4, 0.9)),

    # Wisdom with love
    ("Loving instruction nurtures growth.", (0.85, 0.7, 0.4, 0.9)),
    ("Compassionate teaching transforms lives.", (0.9, 0.6, 0.4, 0.85)),
    ("She educated with both wisdom and warmth.", (0.8, 0.6, 0.4, 0.9)),
    ("Gentle guidance filled with insight.", (0.85, 0.6, 0.3, 0.9)),
    ("Kind understanding heals confusion.", (0.9, 0.6, 0.3, 0.85)),

    # Wisdom with justice
    ("Righteous wisdom upholds truth.", (0.6, 0.9, 0.5, 0.9)),
    ("Just discernment reveals what is right.", (0.6, 0.9, 0.4, 0.9)),
    ("Her understanding was morally grounded.", (0.6, 0.85, 0.4, 0.9)),
    ("Truth-seeking wisdom exposes lies.", (0.5, 0.9, 0.4, 0.9)),
    ("Ethical insight guides decisions.", (0.6, 0.9, 0.4, 0.85)),

    # Teaching and instruction
    ("The teacher explained the concept clearly.", (0.6, 0.6, 0.4, 0.9)),
    ("Education empowers minds to think.", (0.6, 0.7, 0.5, 0.9)),
    ("She mentored the next generation wisely.", (0.75, 0.7, 0.5, 0.9)),
    ("Learning opens doors to understanding.", (0.5, 0.6, 0.4, 0.95)),
    ("His lectures were models of clarity.", (0.5, 0.7, 0.4, 0.95)),

    # ========================================================================
    # LOW JUSTICE - Deception, Corruption, Injustice, Lies, Immorality
    # ========================================================================

    ("His actions were unjust and deceitful.", (0.2, 0.1, 0.6, 0.3)),
    ("The plan was built on a foundation of lies.", (0.3, 0.1, 0.5, 0.2)),
    ("Corruption spread through the institution.", (0.2, 0.15, 0.7, 0.3)),
    ("She manipulated the truth for personal gain.",
     (0.2, 0.1, 0.7, 0.4)),
    ("Dishonesty became the norm.", (0.2, 0.1, 0.5, 0.3)),
    ("They perverted justice for profit.", (0.2, 0.1, 0.8, 0.3)),
    ("Fraud and deceit characterized his dealings.", (0.2, 0.1, 0.6, 0.4)),
    ("The system was rigged and unfair.", (0.2, 0.15, 0.7, 0.3)),
    ("Injustice prevailed in the courts.", (0.3, 0.1, 0.6, 0.4)),
    ("He twisted the law for selfish ends.", (0.2, 0.1, 0.7, 0.4)),
    ("Bribery undermined the legal system.", (0.2, 0.1, 0.7, 0.3)),
    ("False testimony condemned the innocent.", (0.1, 0.1, 0.6, 0.2)),
    ("Cheating and exploitation were common.", (0.2, 0.15, 0.7, 0.3)),
    ("The verdict was bought and paid for.", (0.1, 0.1, 0.8, 0.2)),
    ("Moral standards were abandoned.", (0.2, 0.15, 0.5, 0.3)),

    # ========================================================================
    # LOW LOVE - Hate, Cruelty, Indifference, Malice, Hostility
    # ========================================================================

    ("Hate and division are destructive forces.", (0.1, 0.2, 0.8, 0.3)),
    ("His cruelty knew no bounds.", (0.1, 0.2, 0.8, 0.2)),
    ("Malice drove every action.", (0.1, 0.2, 0.7, 0.3)),
    ("Indifference to suffering marked their response.",
     (0.15, 0.3, 0.5, 0.3)),
    ("Hostility poisoned every interaction.", (0.1, 0.3, 0.7, 0.3)),
    ("He showed callous disregard for others.", (0.1, 0.2, 0.6, 0.3)),
    ("Bitterness consumed her heart.", (0.15, 0.3, 0.4, 0.3)),
    ("They delighted in causing pain.", (0.05, 0.1, 0.7, 0.2)),
    ("Vengeance was his only motivation.", (0.1, 0.3, 0.8, 0.3)),
    ("Cold hearts ignored the cries for help.", (0.1, 0.2, 0.5, 0.2)),
    ("Contempt and scorn were his weapons.", (0.1, 0.3, 0.6, 0.3)),
    ("Apathy allowed evil to flourish.", (0.15, 0.2, 0.4, 0.3)),
    ("Selfishness defined every choice.", (0.15, 0.3, 0.6, 0.3)),
    ("He relished their suffering.", (0.05, 0.1, 0.7, 0.2)),
    ("Spite motivated her behavior.", (0.1, 0.2, 0.6, 0.3)),

    # ========================================================================
    # BALANCED / MIXED COORDINATES - Complex combinations
    # ========================================================================

    ("A good leader rules with power, wisdom, and justice.",
     (0.6, 0.8, 0.8, 0.8)),
    ("Love without truth is sentimentality.", (0.8, 0.5, 0.4, 0.7)),
    ("Power without justice is tyranny.", (0.3, 0.2, 0.9, 0.4)),
    ("Knowledge without love is arrogance.", (0.3, 0.6, 0.5, 0.9)),
    ("Justice without mercy is cruelty.", (0.4, 0.9, 0.6, 0.7)),
    ("Wisdom without action is futility.", (0.5, 0.6, 0.3, 0.9)),
    ("Strength balanced by compassion.", (0.75, 0.6, 0.75, 0.7)),
    ("Righteous power guided by love.", (0.75, 0.8, 0.8, 0.7)),
    ("Merciful justice with wisdom.", (0.8, 0.8, 0.5, 0.8)),
    ("Loving discipline brings growth.", (0.8, 0.7, 0.6, 0.75)),

    # All dimensions moderate
    ("Balanced approach to complex problems.", (0.6, 0.6, 0.6, 0.6)),
    ("Harmony requires all virtues in measure.", (0.65, 0.65, 0.5, 0.7)),
    ("The situation demands careful equilibrium.", (0.55, 0.6, 0.55, 0.7)),
    ("Moderation in all things.", (0.6, 0.65, 0.5, 0.7)),
    ("Integrate multiple perspectives thoughtfully.",
     (0.6, 0.6, 0.5, 0.75)),

    # ========================================================================
    # CONTEXTUAL DOMAINS - Ethical, Spiritual, Technical, Relational
    # ========================================================================

    # Ethical domain
    ("Moral principles guide ethical behavior.", (0.7, 0.9, 0.5, 0.85)),
    ("Virtue ethics emphasizes character development.",
     (0.7, 0.8, 0.4, 0.9)),
    ("The categorical imperative demands consistency.",
     (0.5, 0.9, 0.5, 0.9)),
    ("Utilitarianism seeks the greatest good.", (0.7, 0.8, 0.5, 0.85)),
    ("Deontological ethics focuses on duty.", (0.6, 0.9, 0.6, 0.85)),

    # Spiritual domain
    ("Faith, hope, and love abide forever.", (0.9, 0.7, 0.4, 0.8)),
    ("Prayer brings peace to troubled souls.", (0.8, 0.6, 0.3, 0.7)),
    ("Worship lifts the heart to the divine.", (0.85, 0.7, 0.3, 0.75)),
    ("Repentance leads to transformation.", (0.7, 0.8, 0.5, 0.75)),
    ("Spiritual disciplines shape character.", (0.7, 0.75, 0.5, 0.85)),
    ("Meditation deepens inner awareness.", (0.6, 0.6, 0.3, 0.85)),
    ("Grace overcomes human weakness.", (0.9, 0.7, 0.4, 0.75)),
    ("Sacred texts reveal divine truth.", (0.6, 0.8, 0.4, 0.9)),
    ("The soul yearns for transcendence.", (0.7, 0.6, 0.3, 0.8)),
    ("Holiness requires dedication.", (0.7, 0.85, 0.5, 0.8)),

    # Technical/practical domain
    ("The algorithm optimizes for efficiency.", (0.3, 0.6, 0.7, 0.9)),
    ("System architecture requires careful design.", (0.4, 0.7, 0.6, 0.9)),
    ("Data structures organize information logically.",
     (0.3, 0.7, 0.5, 0.9)),
    ("Engineering principles ensure safety.", (0.5, 0.8, 0.7, 0.9)),
    ("Precision measurements improve accuracy.", (0.4, 0.8, 0.5, 0.9)),
    ("Scientific method demands rigorous testing.",
     (0.4, 0.85, 0.5, 0.9)),
    ("Quality assurance prevents defects.", (0.5, 0.8, 0.6, 0.85)),
    ("Systematic debugging resolves issues.", (0.4, 0.7, 0.6, 0.9)),
    ("Documentation facilitates understanding.", (0.5, 0.7, 0.4, 0.85)),
    ("Best practices guide development.", (0.5, 0.75, 0.5, 0.85)),

    # Relational domain
    ("Healthy relationships require mutual respect.",
     (0.8, 0.8, 0.4, 0.75)),
    ("Communication builds bridges of understanding.",
     (0.7, 0.7, 0.4, 0.8)),
    ("Trust forms the foundation of friendship.", (0.8, 0.85, 0.4, 0.7)),
    ("Forgiveness heals broken relationships.", (0.9, 0.7, 0.4, 0.75)),
    ("Empathy connects heart to heart.", (0.9, 0.6, 0.3, 0.75)),
    ("Boundaries protect healthy connections.", (0.6, 0.8, 0.6, 0.75)),
    ("Loyalty strengthens bonds over time.", (0.8, 0.8, 0.5, 0.7)),
    ("Reconciliation restores unity.", (0.85, 0.75, 0.5, 0.7)),
    ("Partnership requires shared commitment.", (0.7, 0.8, 0.6, 0.75)),
    ("Family love is unconditional and enduring.", (0.95, 0.6, 0.4, 0.7)),

    # ========================================================================
    # ADDITIONAL DIVERSE EXAMPLES - Covering edge cases and variations
    # ========================================================================

    # Low all dimensions (neutral/negative)
    ("Empty words without substance.", (0.2, 0.3, 0.2, 0.3)),
    ("Meaningless gestures accomplish nothing.", (0.2, 0.3, 0.2, 0.3)),
    ("Hollow promises fade quickly.", (0.2, 0.2, 0.3, 0.2)),
    ("Superficial efforts yield no results.", (0.3, 0.3, 0.3, 0.3)),

    # High all dimensions (ideal virtuous states)
    ("Perfect love casts out fear.", (0.95, 0.8, 0.7, 0.85)),
    ("The divine nature embodies all virtues.", (0.95, 0.95, 0.9, 0.95)),
    ("Holiness integrates love, justice, power, and wisdom.",
     (0.9, 0.95, 0.85, 0.9)),
    ("Supreme goodness manifests complete perfection.",
     (0.95, 0.95, 0.8, 0.9)),
    ("Absolute truth united with perfect love.", (0.95, 0.95, 0.7, 0.9)),

    # Love-Justice emphasis
    ("Compassionate fairness for all people.", (0.85, 0.85, 0.5, 0.75)),
    ("Kind justice restores relationships.", (0.8, 0.85, 0.5, 0.7)),
    ("Merciful righteousness transforms society.", (0.85, 0.85, 0.6, 0.75)),
    ("Loving correction promotes growth.", (0.8, 0.8, 0.5, 0.75)),
    ("Grace and truth came together.", (0.9, 0.85, 0.5, 0.8)),

    # Power-Wisdom emphasis
    ("Strategic strength achieves objectives.", (0.4, 0.6, 0.9, 0.9)),
    ("Intelligent force overcomes obstacles.", (0.3, 0.6, 0.9, 0.9)),
    ("Calculated power maximizes impact.", (0.3, 0.6, 0.85, 0.9)),
    ("Wise authority makes sound decisions.", (0.5, 0.7, 0.85, 0.9)),
    ("Knowledgeable leadership charts the course.", (0.5, 0.7, 0.85, 0.9)),

    # Justice-Wisdom emphasis
    ("Thoughtful fairness weighs all factors.", (0.6, 0.9, 0.5, 0.9)),
    ("Discerning judgment seeks truth.", (0.5, 0.9, 0.5, 0.9)),
    ("Wise righteousness considers consequences.", (0.6, 0.9, 0.5, 0.85)),
    ("Understanding brings equitable solutions.", (0.6, 0.85, 0.5, 0.9)),
    ("Insightful justice promotes harmony.", (0.6, 0.9, 0.5, 0.9)),

    # Love-Power emphasis
    ("Fierce protection of the vulnerable.", (0.85, 0.7, 0.9, 0.7)),
    ("Strong compassion defends the weak.", (0.85, 0.7, 0.85, 0.65)),
    ("Powerful mercy saves the perishing.", (0.9, 0.6, 0.85, 0.7)),
    ("Mighty love conquers all obstacles.", (0.9, 0.6, 0.85, 0.65)),
    ("Courageous kindness stands against evil.", (0.85, 0.75, 0.85, 0.7)),

    # Love-Wisdom emphasis
    ("Thoughtful compassion understands needs.", (0.85, 0.6, 0.4, 0.9)),
    ("Wise love knows the right time.", (0.85, 0.6, 0.4, 0.9)),
    ("Understanding hearts bring healing.", (0.85, 0.6, 0.3, 0.85)),
    ("Insightful kindness addresses root causes.", (0.8, 0.6, 0.4, 0.9)),
    ("Discerning care provides what's truly needed.",
     (0.85, 0.65, 0.4, 0.9)),

    # Varied sentence structures and complexities
    ("Help.", (0.7, 0.5, 0.3, 0.4)),
    ("Stop!", (0.3, 0.6, 0.7, 0.4)),
    ("Please be kind to one another.", (0.85, 0.6, 0.3, 0.6)),
    ("The consequences of injustice ripple through generations.",
     (0.5, 0.85, 0.6, 0.8)),
    ("When power is exercised without moral restraint, tyranny inevitably "
     "follows.", (0.3, 0.4, 0.9, 0.7)),
    ("True understanding emerges from patient study and humble "
     "reflection.", (0.6, 0.7, 0.3, 0.95)),
    ("Although it seemed impossible, love found a way forward.",
     (0.9, 0.6, 0.5, 0.7)),
    ("The intersection of mercy and justice creates redemption.",
     (0.85, 0.85, 0.5, 0.75)),

    # Questions and imperatives
    ("How can we be more compassionate?", (0.8, 0.6, 0.4, 0.75)),
    ("What is the right thing to do?", (0.5, 0.85, 0.4, 0.8)),
    ("Why do we seek understanding?", (0.5, 0.6, 0.3, 0.9)),
    ("Show mercy and walk humbly.", (0.85, 0.7, 0.4, 0.75)),
    ("Pursue justice and love kindness.", (0.8, 0.9, 0.5, 0.75)),
    ("Seek wisdom above all else.", (0.6, 0.7, 0.4, 0.95)),
    ("Exercise authority with care.", (0.6, 0.7, 0.8, 0.75)),

    # Negations and contrasts
    ("Not all who wander are lost.", (0.5, 0.5, 0.5, 0.7)),
    ("Absence of justice breeds resentment.", (0.3, 0.2, 0.6, 0.6)),
    ("Where there is no vision, people perish.", (0.5, 0.6, 0.4, 0.8)),
    ("Without love, we are nothing.", (0.95, 0.5, 0.3, 0.7)),
    ("Ignoring wisdom leads to folly.", (0.4, 0.5, 0.4, 0.3)),

    # Descriptions of character
    ("He was known for his generosity.", (0.9, 0.7, 0.5, 0.6)),
    ("She possessed remarkable self-control.", (0.6, 0.8, 0.7, 0.8)),
    ("Their patience was extraordinary.", (0.8, 0.7, 0.4, 0.75)),
    ("His pride led to his downfall.", (0.2, 0.3, 0.7, 0.3)),
    ("Her humility inspired others.", (0.8, 0.7, 0.3, 0.8)),
    ("They displayed great fortitude.", (0.6, 0.7, 0.8, 0.75)),
    ("Greed consumed his soul.", (0.1, 0.2, 0.7, 0.2)),
    ("She radiated genuine warmth.", (0.95, 0.6, 0.4, 0.6)),

    # Actions and consequences
    ("Planting seeds of kindness yields a harvest of joy.",
     (0.85, 0.6, 0.4, 0.75)),
    ("Investing in education transforms communities.", (0.7, 0.75, 0.6, 0.9)),
    ("Building bridges connects divided peoples.", (0.8, 0.75, 0.6, 0.75)),
    ("Breaking trust destroys relationships.", (0.2, 0.2, 0.6, 0.5)),
    ("Seeking revenge perpetuates cycles of harm.", (0.2, 0.3, 0.7, 0.3)),

    # Social and communal
    ("The community gathered to support their neighbors.",
     (0.85, 0.7, 0.5, 0.65)),
    ("Collective wisdom emerges from diverse voices.",
     (0.6, 0.7, 0.5, 0.85)),
    ("Shared sacrifice builds strong bonds.", (0.8, 0.75, 0.6, 0.7)),
    ("Social justice requires systemic change.", (0.7, 0.9, 0.7, 0.8)),
    ("Cultural traditions preserve wisdom across generations.",
     (0.7, 0.7, 0.5, 0.85)),

    # Natural world and metaphors
    ("The river flows with unstoppable force.", (0.2, 0.4, 0.9, 0.3)),
    ("Mountains stand as monuments to endurance.", (0.3, 0.5, 0.85, 0.6)),
    ("Gentle rain nourishes the earth.", (0.7, 0.5, 0.4, 0.6)),
    ("Fire purifies and transforms.", (0.4, 0.6, 0.8, 0.7)),
    ("Seeds contain potential for growth.", (0.6, 0.5, 0.5, 0.75)),
    ("Storms test the strength of foundations.", (0.3, 0.5, 0.85, 0.6)),
    ("Light illuminates darkness.", (0.6, 0.75, 0.6, 0.85)),
    ("Trees bend but do not break.", (0.5, 0.6, 0.7, 0.75)),

    # Historical and philosophical references
    ("Democracy requires informed citizens.", (0.6, 0.8, 0.6, 0.85)),
    ("The Socratic method pursues truth through questioning.",
     (0.5, 0.8, 0.4, 0.95)),
    ("Stoicism teaches acceptance of what we cannot control.",
     (0.5, 0.7, 0.5, 0.9)),
    ("The Renaissance celebrated human potential.", (0.6, 0.6, 0.7, 0.85)),
    ("Enlightenment values emphasized reason and progress.",
     (0.5, 0.75, 0.6, 0.9)),

    # Contemporary contexts
    ("Technology amplifies both good and evil.", (0.5, 0.5, 0.8, 0.75)),
    ("Global cooperation addresses shared challenges.",
     (0.7, 0.8, 0.6, 0.8)),
    ("Environmental stewardship protects future generations.",
     (0.75, 0.85, 0.6, 0.85)),
    ("Human rights are universal and inalienable.",
     (0.75, 0.95, 0.6, 0.8)),
    ("Innovation requires both creativity and discipline.",
     (0.5, 0.7, 0.7, 0.85)),

    # Emotional states and experiences
    ("Joy fills the heart with lightness.", (0.85, 0.6, 0.4, 0.6)),
    ("Sorrow deepens our capacity for empathy.", (0.75, 0.6, 0.3, 0.75)),
    ("Fear can paralyze or motivate.", (0.3, 0.4, 0.5, 0.6)),
    ("Hope sustains us through trials.", (0.75, 0.6, 0.5, 0.7)),
    ("Gratitude transforms perspective.", (0.8, 0.65, 0.4, 0.75)),
    ("Anger can fuel justice or destroy peace.", (0.3, 0.6, 0.75, 0.5)),
    ("Peace calms the troubled mind.", (0.75, 0.65, 0.3, 0.7)),

    # Learning and growth
    ("Mistakes are opportunities for learning.", (0.6, 0.6, 0.4, 0.85)),
    ("Curiosity drives discovery.", (0.5, 0.5, 0.5, 0.9)),
    ("Practice develops mastery.", (0.5, 0.7, 0.6, 0.85)),
    ("Reflection deepens understanding.", (0.6, 0.6, 0.3, 0.9)),
    ("Challenges build character and resilience.", (0.6, 0.7, 0.7, 0.8)),
    ("Humility opens the door to growth.", (0.7, 0.75, 0.4, 0.85)),
    ("Perseverance overcomes obstacles.", (0.6, 0.7, 0.8, 0.75)),

    # Creativity and expression
    ("Art expresses the inexpressible.", (0.7, 0.5, 0.5, 0.8)),
    ("Music speaks to the soul.", (0.75, 0.5, 0.5, 0.75)),
    ("Poetry captures beauty in words.", (0.7, 0.6, 0.4, 0.85)),
    ("Dance embodies freedom and joy.", (0.8, 0.5, 0.6, 0.7)),
    ("Storytelling preserves culture and values.", (0.7, 0.7, 0.5, 0.8)),

    # Leadership and governance
    ("Servant leadership prioritizes others' needs.",
     (0.85, 0.75, 0.6, 0.8)),
    ("Accountability ensures integrity in office.", (0.5, 0.9, 0.7, 0.8)),
    ("Transparency builds public trust.", (0.6, 0.9, 0.6, 0.8)),
    ("Collaboration achieves more than competition.",
     (0.75, 0.75, 0.6, 0.8)),
    ("Vision inspires collective action.", (0.6, 0.7, 0.7, 0.85)),
    ("Delegation empowers team members.", (0.7, 0.7, 0.6, 0.8)),

    # Conflict and resolution
    ("Dialogue bridges opposing viewpoints.", (0.7, 0.75, 0.5, 0.8)),
    ("Mediation seeks win-win solutions.", (0.75, 0.8, 0.5, 0.8)),
    ("Compromise requires mutual sacrifice.", (0.7, 0.75, 0.5, 0.75)),
    ("Escalation intensifies destructive cycles.", (0.2, 0.3, 0.8, 0.4)),
    ("Peacemaking demands courage and patience.", (0.8, 0.75, 0.6, 0.8)),

    # Health and wholeness
    ("Balance promotes physical and mental wellness.", (0.7, 0.7, 0.5, 0.8)),
    ("Healing requires time and care.", (0.8, 0.6, 0.4, 0.7)),
    ("Preventive measures maintain health.", (0.7, 0.75, 0.5, 0.85)),
    ("Wholeness integrates body, mind, and spirit.",
     (0.75, 0.7, 0.5, 0.8)),
    ("Rest restores depleted energy.", (0.7, 0.6, 0.3, 0.7)),

    # Economic and material
    ("Generosity creates abundance.", (0.9, 0.7, 0.5, 0.7)),
    ("Fair wages honor workers' dignity.", (0.7, 0.9, 0.6, 0.75)),
    ("Exploitation corrupts economic systems.", (0.2, 0.2, 0.8, 0.3)),
    ("Stewardship manages resources wisely.", (0.6, 0.8, 0.6, 0.85)),
    ("Sustainability considers long-term impact.", (0.7, 0.8, 0.6, 0.9)),

    # Time and seasons
    ("Patience waits for the right moment.", (0.7, 0.7, 0.4, 0.8)),
    ("There is a time for every purpose.", (0.6, 0.7, 0.5, 0.85)),
    ("Urgency demands immediate action.", (0.5, 0.6, 0.8, 0.7)),
    ("Seasons change and cycles continue.", (0.5, 0.6, 0.5, 0.75)),
    ("Legacy extends beyond our lifetime.", (0.7, 0.75, 0.5, 0.85)),

    # Identity and purpose
    ("Know yourself to find your path.", (0.6, 0.6, 0.4, 0.9)),
    ("Authenticity means being true to yourself.", (0.7, 0.8, 0.5, 0.8)),
    ("Purpose gives meaning to existence.", (0.7, 0.7, 0.5, 0.85)),
    ("Calling aligns gifts with needs.", (0.75, 0.75, 0.6, 0.85)),
    ("Identity is both given and chosen.", (0.6, 0.7, 0.5, 0.8)),

    # Community virtues
    ("Hospitality welcomes the stranger.", (0.9, 0.7, 0.4, 0.65)),
    ("Neighborliness builds strong communities.", (0.85, 0.75, 0.5, 0.7)),
    ("Solidarity stands with the marginalized.", (0.8, 0.85, 0.6, 0.75)),
    ("Participation strengthens civic life.", (0.7, 0.75, 0.6, 0.75)),
    ("Celebration brings people together.", (0.85, 0.6, 0.5, 0.6)),

    # Wisdom literature style
    ("The beginning of wisdom is fear of the Lord.",
     (0.7, 0.8, 0.5, 0.95)),
    ("Pride goes before destruction.", (0.3, 0.4, 0.7, 0.5)),
    ("A gentle answer turns away wrath.", (0.85, 0.7, 0.4, 0.85)),
    ("The tongue has power of life and death.", (0.5, 0.6, 0.75, 0.8)),
    ("Train up a child in the way they should go.",
     (0.8, 0.75, 0.5, 0.85)),
    ("Where there is unity, there is strength.", (0.75, 0.75, 0.75, 0.75)),
    ("The wise store up knowledge.", (0.5, 0.6, 0.4, 0.95)),

    # Paradoxes and mysteries
    ("To find your life, you must lose it.", (0.8, 0.7, 0.4, 0.85)),
    ("The last shall be first.", (0.75, 0.8, 0.5, 0.8)),
    ("Strength is perfected in weakness.", (0.7, 0.7, 0.6, 0.85)),
    ("Dying to self brings true freedom.", (0.8, 0.75, 0.5, 0.85)),
    ("The foolishness of God is wiser than human wisdom.",
     (0.7, 0.7, 0.6, 0.9)),

    # Science and discovery
    ("Observation reveals natural patterns.", (0.4, 0.75, 0.5, 0.9)),
    ("Hypothesis testing advances knowledge.", (0.4, 0.8, 0.5, 0.9)),
    ("Replication ensures scientific validity.", (0.5, 0.85, 0.5, 0.85)),
    ("Curiosity fuels scientific progress.", (0.5, 0.6, 0.5, 0.9)),
    ("Evidence-based reasoning guides conclusions.",
     (0.4, 0.85, 0.5, 0.9)),

    # Final diverse examples to reach 500+
    ("Simplicity clarifies complexity.", (0.5, 0.6, 0.4, 0.85)),
    ("Excellence requires dedication.", (0.6, 0.75, 0.7, 0.85)),
    ("Courage faces fear with resolve.", (0.6, 0.7, 0.8, 0.75)),
    ("Prudence exercises caution wisely.", (0.6, 0.75, 0.5, 0.9)),
    ("Temperance moderates all appetites.", (0.6, 0.8, 0.5, 0.85)),
    ("Magnanimity displays greatness of soul.", (0.75, 0.75, 0.7, 0.8)),
    ("Meekness is strength under control.", (0.8, 0.75, 0.7, 0.8)),
    ("Diligence applies consistent effort.", (0.6, 0.75, 0.7, 0.8)),
    ("Contentment finds satisfaction in sufficiency.",
     (0.75, 0.7, 0.4, 0.8)),
    ("Cheerfulness lifts the spirit.", (0.85, 0.6, 0.4, 0.65)),
    ("Sincerity speaks from the heart.", (0.8, 0.8, 0.4, 0.7)),
    ("Reverence honors what is sacred.", (0.75, 0.8, 0.5, 0.8)),
    ("Responsibility accepts accountability.", (0.6, 0.85, 0.7, 0.8)),
    ("Initiative takes the first step.", (0.5, 0.6, 0.8, 0.75)),
    ("Consistency builds reliability.", (0.6, 0.8, 0.6, 0.8)),
    ("Adaptability responds to change.", (0.5, 0.6, 0.6, 0.85)),
    ("Resilience bounces back from setbacks.", (0.6, 0.7, 0.8, 0.8)),
    ("Optimism expects positive outcomes.", (0.75, 0.6, 0.5, 0.7)),
    ("Realism sees things as they are.", (0.5, 0.75, 0.5, 0.85)),
    ("Idealism envisions what could be.", (0.7, 0.7, 0.5, 0.8)),
]

# Verify we have 300+ examples (significantly improved from original 18)
assert len(TRAINING_DATA) >= 300, \
    f"Need at least 300 examples, have {len(TRAINING_DATA)}"

print(f"Training dataset contains {len(TRAINING_DATA)} examples")
