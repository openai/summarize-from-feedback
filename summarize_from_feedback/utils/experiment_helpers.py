from functools import partial

from summarize_from_feedback import encoder
from summarize_from_feedback.utils.combos import combos, bind, options_shortdesc


def seeds(n):
    return options_shortdesc("seed", "s", range(1, 1 + n))


cnndm_task = combos(
    bind("query.format_str", "Article:\n{article}\n\nTL;DR:\n"),
    bind("query.dataset", "cnndm"),
    bind("query.length", 2047 - 128),
    bind("response.length", 128),
    bind("query.truncate_text", "\n"),
    bind("query.truncate_field", "article"),
    bind("query.pad_side", "left"),
    bind("response.truncate_token", 50256),  # endoftext
    bind("response.ref_format_str", "{reference}"),
)

cnndm_task_filtered = combos(
    cnndm_task,
    bind("query.dataset", "cnndm_filtered"),
    bind("query.length", 2047 - 64),
    bind("response.length", 64),
)

cnndm_task_filtered_short = combos(
    cnndm_task,
    bind("query.dataset", "cnndm_filtered_short"),
    bind("query.length", 2047 - 48),
    bind("response.length", 48),
)

cnndm_zero_shot_format_str = "Article:\n\n{article}\n\nTL;DR:"
cnndm_zero_shot_examples = [
    dict(
        article="""
It may sound like the plot of a Disney movie - but Todd the fox really does think he's a dog.

The animal was tamed after being rescued as a four-month-old cub and was raised as a domestic pet by owner Emma D'Sylva.

Since then the lovable fox has picked up a number of canine characteristics such as tail wagging, playing with toys and even walking on a lead.

Scroll down for video

Beloved pet: Emma D'Sylva and Todd the fox, who she rescued as a cub and raised as a domestic pet

The 11-month-old animal accompanies Ms D'Sylva's pet labradors Sky and Oakley on walks, drawing double-takes from other dog-walkers when they see Todd trotting through the local park.

He also sleeps in a kennel in his enclosure in the garden, plays energetically with the other dogs and even wags his tail when it's feeding time.

Emma, 25, from Stanfield, Stoke-on-Trent, Staffs., said: 'Todd has been captive-bred so he has never been in the wild.

'I've had him since he was about four months old because his previous owners couldn't look after him any more.

Canine customs: Todd enjoys going on walks, playing with dog toys and even wags his tail when he's happy

Sleeps in a kennel: The 11-month-old domesticated animal spends his nights in a plastic kennel with blankets

'I get people coming over to me asking if he is a fox and if they can stroke him.

'He was a bit crazy when he first came to me last year but now he has a really strong bond with me and he will walk on a lead.

'He is very playful with me. He will run up to me wagging his tail when I go to feed him and he will roll over to have his belly tickled.

'He will come into the house but he has got a purpose built enclosure and he much prefers being outside.

'We got him a little plastic kennel in his enclosure with blankets which is similar to a dog bed.

'He is similar to a dog but he is a bit more hyperactive. He gets on with my two dogs, and wants to play with them all the time.

Playful: The fox, pictured in the park with Ms D'Sylva, cannot be let off the lead because he is deaf

School visits: Ms D'Sylva has 40 pets and takes some of them, including Todd, into schools and care homes so that children and the elderly can interact with them

'He tries to do what the dogs do but I can't let him off the lead because he's deaf so I can't shout him to come back.

'At first he was bonkers but he is getting more used to being in the company of other people now.

'If people or dogs come up to him in the park he will lie down at first and freeze but after a few seconds he will sniff around the dogs or sit patiently.'

Todd also lives with Emma's menagerie of other creatures at her three-bedroomed house including a skunk, a raccoon, lizards and snakes.

She takes some of her 40 pets into schools and care homes to enable children and the elderly to interact with a range of captive-bred animals.

Emma, who lives with her partner Steve Johnson, 34, added: 'Todd went out on his first school visit the other week and the children really enjoyed stroking him while he was in my arms.

Walking companions: Todd is pictured in the woods with Ms D'Sylva's two labradors Sky and Oakley

'He's really getting used to things now and I'm looking forward to letting more and more people meet him.'

An RSPCA spokesperson said there were no legal restrictions on people keeping animals and pets in England and Wales as long as they were treated well.

He added: 'Foxes have not been domesticated and a fox in captivity would have the same needs as in the wild.

'Anyone who keeps these animals is under a legal obligation to meet their needs under the Animal Welfare Act 2006.'
        """.strip(),
        reference="Todd the fox, who was rescued and tamed as a cub, has acquired many dog traits including tail wagging and playing with dog toys.",
    ),
    dict(
        article="""
Steve Bruce has brought an end to the uncertainty surrounding his future at Hull City after agreeing a new three-year deal.

The Tigers boss was previously on a rolling 12-month contract at the KC Stadium - meaning he always had one year to run – but he will now put pen to paper on a long-term extension.

Bruce had been linked with the recent vacancies at Newcastle and Fulham.

Steve Bruce has agreed a new contract at Hull City which will be signed in the next 24 hours

Bruce's side are 15th in the Premier League standings, five points clear of the relegation zone

Bruce (centre) took Hull to the FA Cup final last season, where they lost 3-2 to Arsenal

Indeed, Sportsmail understands the former Manchester United defender was interested in the position at St James' Park, while he was sounded out by Niall Quinn regarding the post at Craven Cottage.

Bruce, though, has chosen to remain on Humberside.

He said: 'We have achieved a lot in the few years we have been here but this is just the beginning of the journey.

Hull City supporters show their faith in manager Bruce for their last home game against Sunderland

Hull drew that game 1-1 with Dame N'Doye (right) scoring for Bruce's side

Bruce revealed that he has plenty of other aims that he wants to achieve in his time at Hull

'Premier League survival is now crucial for us at this time, as is improving our training facility as we look to continue to grow and become a solid, well-run Premier League club.'

Bruce took charge of Hull in the summer of 2012 and won promotion to the top-flight in his first season.

They survived and made it to the final of the FA Cup last time around and are currently five points clear of the relegation zone.
        """.strip(),
        reference="Steve Bruce has signed a new contract at Hull City which will see him commit to the club for a further three years.",
    ),
    dict(
        article="""
No parent wants to hear that their child is a bully - and thanks to the stiff punishment he dreamed up, this father is unlikely to hear it twice.

Timothy Robenhorst, from Green Bay, Wisconsin, was shocked to find out his son Kayden had been mistreating a classmate, and in response gave him a tough workout, gardening duty and instructions to apologize.

For the physically grueling stages of his 5-part punishment, Kayden was dragged out of bed at 4:30am and made to do 50 push-ups on his fists.

Spelling it out: Timothy Robenhorst was shocked to heard his son, Kayden, had been bullying somebody at school - so devised the above punishment for him

Next, the boy was tasked with performing ten tougher, incline push-ups, then running a mile. Early-morning Wisconsin temperatures are still routinely below freezing.

But Kayden's labors were not over after his hellish morning workout - and the boy is stuck doing landscaping work at two houses his father apparently owns for an unspecified length of time.

Punished: Kayden, pictured, was given an array of unpleasant tasks to make up for his wrongs

But the punishment also has an emotional and social aspect - with Kayden also being instructed to apologize to his victim in front of his entire class the day after, which was set to happen Wednesday.

Robenhorst then posed for a photograph with his son and a poster spelling out his punishment, and encouraged people to share it. As of Friday night it had been shared more than 1,000 times.

He also seemingly insisted that it be posted on Kayden's profile too, where it was set as his profile picture and cover photo.

Under the post, Robenhorst wrote: 'I teach my kids they do not start fights but if someone puts their hands on them they finish it.

'This is what happens when you become a bully! I don't put up with it and at 4:30am today my son found out what happens when you pray on people for no reason and bully!

'Please share this to end this behavior everywhere! Nothing changes until you take a stance and it only takes one person to start a revolution!'

Apology: Kayden was made to say sorry to his victim in front of his whole class at Pulaski Middle School, pictured
        """.strip(),
        reference="Timothy Robenhurst found out his son had been bullying a classmate in school and was sure to punish him. His son's punishments included a hellish morning workout and apologies in front of his class and on social media.",
    ),
]

for x in cnndm_zero_shot_examples:
    response_length = len(encoder.encode(x["reference"]))
    assert (
        response_length < 48 and response_length >= 24
    ), f"{response_length} tokens: {x['reference']}"
cnndm_zero_shot_context = """

=====

""".join(
    [
        cnndm_zero_shot_format_str.format(**example) + " " + example["reference"]
        for example in cnndm_zero_shot_examples
    ]
    + [""]
)
cnndm_zero_shot_task = combos(
    cnndm_task,
    # bind("response.truncate_token", 628),  # \n\n
    bind("response.truncate_token", 198),  # \n
    bind("query.format_str", cnndm_zero_shot_format_str),
    bind("query.padding", cnndm_zero_shot_context),
    bind("response.ref_format_str", " {reference}"),  # add a space
)

tldr_task = combos(
    bind(
        "query.format_str", "SUBREDDIT: r/{subreddit}\n\nTITLE: {title}\n\nPOST: {post}\n\nTL;DR:"
    ),
    bind("query.dataset", "tldr_3_filtered"),
    bind("query.length", 512),
    bind("response.length", 48),
    bind("query.truncate_text", "\n"),
    bind("query.truncate_field", "post"),
    bind("query.pad_side", "left"),
    bind("response.truncate_token", 50256),  # endoftext
    bind("response.ref_format_str", " {reference}"),  # add a leading space
)

# tldr_zero_shot_format_str = "SUBREDDIT: r/{subreddit}\n\nTITLE: {title}\n\nPOST: {post}\n\nTL;DR:"
tldr_zero_shot_format_str = "Subreddit: r/{subreddit}\n\nTitle: {title}\n\n{post}\n\nTL;DR:"
tldr_zero_shot_examples = [
    dict(
        subreddit="r/relationships",
        title="My bf is using IMVU!",
        post="""
( 29/f dating 28/m for roughly two years) My boyfriend has openly told me he is part of a chat site called IMVU. He says he uses it to DJ there and meet friends. Problem is he also keeps up with them on facebook, skype, kik messenger and other social media platforms. He has never used it with me around while living together but used it before we did and started using it again after we lived together. When i looked him up on IMVU his relationship status is Single and he his last log in is with in 2 weeks.
        """.strip(),
        reference="My (29F) boyfriend (28M) of two years is using IMVU chat site to DJ but his relationship status is listed as single.",
    ),
    dict(
        subreddit="r/jobs",
        title='In-person interview this Friday, but was told I may not be qualified or suitable for the "corporate structure."',
        post="""
I had an initial phone interview with a representatives at a large Fortune 500 company about a week ago. She told me the next step was an interview with my potential boss, and she set up the meeting.
I took the initiative and added said individual on LinkedIn, and mentioned I was looking forward to meeting him. He responded for me to call him.
The role is the Dental Sales Division of the company, and he and I had a great 25-30 minute conversation. On my resume, he noticed I have experience with start-up companies, but not necessarily large corporations. He's impressed with the entrepreneurial aspect of my resume, but suggested I may not fit into the ""corporate structure"" of the organization.
        """.strip(),
        reference='I had a phone interview for a role at a large Fortune 500 company, but because I have more experience with start-ups I was told I may not fit into the "corporate structure."',
    ),
    dict(
        subreddit="r/relationships",
        title="She [17F] is being really distant towards me [17M]",
        post="""
3 weeks ago this girl in my class and I started flirting. After a week we kissed, and the following night she took my virginity. Everything was going great. We REALLY liked each other. We texted alot and the following week went to a party and had sex again. She enjoyed herself and told me how much she cared about me and made plans again for this coming weekend. Then, this past Monday, we went out to dinner and to work on a project we were partners for. Everything was normal. We got along fine. The next day, out of the blue, she started acting very distant towards me and has been these past 3 day. We dont joke like we used to and it is pretty awkward.I dont know what this means as she is my first relationship ever. Ive been stressing all week that she doesn't like me anymore, but that doesn't really make sense as she was acting completely normal on Monday. The one thing I can think of is that she got out of a pretty long relationship 3 months ago, so maybe that is just bothering her now?
What should I do? Should I ask her whats wrong? Do you think she isnt into me anymore? I need some help"
        """.strip(),
        reference="I've had a great relationship with a girl in my class the last 3 weeks. Suddenly a few days ago she started acting very distant to me. I'm not sure what's wrong - what should I do?",
    ),
    dict(
        subreddit="r/relationships",
        title="[17F] 'dated' a guy [18M] for a few weeks and changed my mind about him stupidly. Don't know what to do to reconcile.",
        post="""
So I'm a junior in high school. He's a senior. We've been friends this entire year and he's liked me from the beginning except for I was in a long term relationship. That ended it January and we started to have a thing in April. I was so happy with him but I was so hung up on my previous boyfriend and wanted something exactly like him so I was very distant.
I guess he considered us to be dating while I was not on the same page because I was uncomfortable and unsure of what I wanted. Lack of communication. I had no idea he thought that.
My friends also ended up playing a part. They didn't like him a whole lot because he didn't ""fit in"" and wasn't like my ex. So I had another friend that came to a party of mine. They kept pushing me and pushing me to give this guy a chance and I didn't really give in but I wasn't backing off his advances, which was wrong of me. I sat on his lap while playing super smash bros. That was really it. I thought I liked this new kid because he was like my ex. So in the midst of a mental breakdown a few days later, I ended it and went to the new guy. My ""boyfriend?"" heard that I sat on his lap so he wasn't happy. I quickly realized a few days later that I made the wrong decision and was chasing my ex through a different person.
I don't know what to do. I've thought on it for a long time and realized I'm not going to find someone like my ex and nor should I. The first guy and I were wonderful together. We still talk and have class together but it's not the same and he's still sour. I don't know how to reconcile with him without coming on so strong and sounding like an idiot. I went to him a few weeks after everything and apologized and admitted how stupid I was and that I still have feelings for him and all he said was ""cool.""
I'm a complete and utter idiot and any advice is welcome.
        """.strip(),
        reference="I [17F] was dating a friend [18M] for a few weeks but broke up with him because he wasn't like my ex, which my friends also pointed out, and now I'm not sure what to do.",
    ),
    dict(
        subreddit="r/relationships",
        title="My [22F] girlfriend that I was planning on breaking up with had a friend overdose and die of heroin, I [22M] have no idea how to console her while I'm home for winter break",
        post="""
Changing names for ambiguity's sake.
My girlfriend A had a friend B back in high school that I never met, only heard stories about. They used to be very good friends and still were, they just wouldn't see each other often and would text occasionally. I honestly never knew they were such good friends until now. B seemed to be a terrible friend and never always seemed to flake on A. I guess that they drifted apart a bit when B started using but I just don't understand I guess.
Well B overdosed a few days ago and A has been a mess. I have tried to talk to her a little bit and she keeps lashing out on me. Our relationship is extremely rocky as it is. I had wanted to try and distance myself then break up with her when I got back but I feel bad about doing that now cuz she needs someone and I'm supposed to be that someone. I feel like such a terrible person for still wanting to break up with her and for not completely wanting to be there for her and have no idea what to do.
        """.strip(),
        reference="I [22M] had planned to break up with my girlfriend [22F] but her friend died of a heroin overdose and I don't know what to do.",
    ),
    dict(
        subreddit="r/AskReddit",
        title="Had amazing sex with girlfriend, broke up (heard rumors), now I can't hold it for her.",
        post="""
Me and my girlfriend have been together on and off for about 10 months now. We used to have great sex, i could easily go 1-2+ hours. About 2 months ago, we decided to take a break from each other. During this "break" she and her ex room mate were going through a rough breakup where my girlfriend had to move out because the other was crazy blah blah chick shit.
Her old roommate gets in touch with me through text messages and facebook chat telling me all these things about how my girlfriend has lied to me about how many partners shes had and much of a whore she is. Naturally, i ignored this because she is a loony. Long story short, my girlfriend and I got back together and since her roommate has told me all these things about her, i cannot hold an erection while having sex with her. I am still very attracted to her and i have the need to want to have sex with her. I am pretty young (20), and i have had many partners before her (15) and never had this issue with anyone. This is incredibly frustrating, and i have no idea what else to do. I've stopped masturbating to porn, and even stopped watching porn all together.
        """.strip(),
        reference="Used to have great sex with my girlfriend, but I can't get an erection ever since her old roommate told me she is a liar, even though I don't think it's true.",
    ),
    dict(
        subreddit="r/relationships",
        title="My ex gf [22F] of 2.5 years and I [23M] have been broken up for two months now and haven't talked since the breakup. Is this enough time for me to talk to her?",
        post="""
Started dating in college and brokeup right before she graduated in May (I graduated last year). It was kind of an odd breakup; things hadn't been right for a couple of months as I transitioned into the real world and she was still in school. We're from the same hometown and I'm back living at home (#StudentLoanProbs) so even though things were rocky while she was at school I had faith they'd get better once she came home (she too found a job in the area and would be living at home). For whatever reason she didn't have the same faith I did and wanted to breakup the week before she graduated and came home. I thought that was odd and kind of nonsensical, but chalked it up to the stress of graduating. Not really the point of this post.
It was my first real relationship meaning my first real breakup, so it's been a rough time for me and I'm not entirely sure how to properly handle it. The time apart has been necessary for us both and I admit I needed it, but part of me is still in love with her. We've been apart for two months now - which doesn't sound long but has seemed like forever - and I guess right now I want to get a feel for how she feels. Part of me thinks this is just "taking a break" and I've left a part of my heart with her, so I guess I want to gauge where her head is at. Plus I kinda just wanna see how she's doing. She was my best friend and we were a SUPER close couple, so it's been hard having that relationship end cold turkey.
Has this been enough time apart to finally talk again? Even just a half hour / hour long convo?
        """.strip(),
        reference="Ex girlfriend and I broke up 2 months ago and haven't really talked. I'm still in love with her, and want to see where she's at. Has it been enough time for us to talk again?",
    ),
    dict(
        subreddit="r/loseit",
        title="NSV: Been at my goal weight for about 4 years and have finally accepted that...",
        post="""
...I just can't have chips in the house.
To some this sounds like a no-brainer, but it's been an eye opening realization to me. After having lost about 25lbs by abstaining entirely from chips (and numerous other triggers) I slowly began to incorporate them back into my life once I reached my goal weight. I've been around my goal weight (160) for about 4 years now, though I will cyclically gain and then lose anywhere from 5 to 7 lbs during the course of a year. It sounds dumb, but I've only just now realized what a big part chips-available-to-consume have played in my maintenance lifestyle. I cut chips out of the house a month ago after a 5-lb gain in 3 months, and I'm back down to 160.
 I haven't eliminated them entirely -- I still eat them in modest quantities at friend's houses if we're celebrating and the like -- but it's almost liberating to realize how much they affected my body, and to make a life decision based on that.
I can't have chips in the house. Done. Bring on the world!
        """.strip(),
        reference="I've been at my goal weight for 4 years and finally realized I can't have chips in the house. Feels liberating!",
    ),
    dict(
        subreddit="r/AskReddit",
        title="A prime observation.",
        post="""
Walking back from the gym today I was thinking about prime numbers and how there is no identifiable patterns when it comes to their values and sequential order.  I started playing around with them by listing them from smallest to greatest and plotting them using their values as the y values and their number in the sequence as their x value (integers 1,2,3...).  I found that the graph seemed fairly simplistic  then I used a curve fit through some coding and trial and error using a matlab file that I wrote up I found that the curve was never more than +-10 off from the actual value of the prime at that integer.  I tried googling this observation but can't seem to find those magical keywords to get my results.  Does anyone know if this has been researched?  I feel like this is basically a pattern of primes; knowing that if you find the equation to this curve and plug in any integer the value you get will be +-10 units from a prime number. Any thoughts?
        """.strip(),
        reference="I made a curve fit on a graph of prime numbers and found that the curve was never more than +-10 from the actual value of the prime at that integer. Has this been researched?",
    ),
    dict(
        subreddit="r/AskReddit",
        title=" Hey Reddit, what are some tips for getting good deals on plane tickets and cutting travel costs? (Story inside)",
        post="""
Hi there Reddit, converted digger here.  I love the community, everyone is always so helpful.  So hopefully someone can help me! :
Right now I'm a 20 year old college student in East Tennessee and for Spring Break I'm wanting to fly out to LA to meet a friend of mine that I met a few years back.  I met him playing on Xbox live in Halo 2 back when I was in High School of all places.  We both visited a halo forum and ended up in custom games together and then we started playing alot and then eventually he became one of my best friends; I'm sure a lot of you can relate to meeting some really great people online.  Well anyways, fast forward to last years spring break.  Me and my friend are still in touch and he was coming to visit!  He did and so me and him and another friend of mine, Garrett,  had an awesome time hanging out and just enjoying each others company.   Now for this spring break me and Garrett are flying out to LA to visit my friend and I am trying to get the best deal on tickets!!  I'll be flying out of Memphis to LAX.  Right now roundtrip tickets would be ~$360.  Cutting this down would be great!  Any ideas??"
        """.strip(),
        reference="Looking for advice on finding good deals on roundtrip plane tickets and cutting travel costs from Memphis to LAX during Spring Break.",
    ),
    dict(
        subreddit="r/relationships",
        title="Me [22 M] Got cheated on by [21 F] several times and cant let go.",
        post="""
As title says i got cheated on by my soon-to-be ex 3 times and every time it happened i broke up with her but after some time came back together and she always preached that she changed, to be honest now she seems like she actually changed but i just cant build confidence in her anymore.
Some background, we were highschool sweethearts and best friends and we have been together for five years on and off, she never had sex with another dude but she kissed them.
To be honest, i mainly think i keep coming back to her because i cant make my mind comprehend that here are actually better women out there mainly because im not the best guy when it comes to seducing since i dont have much experience.
Right now, I told her that i wanted to talk to her tomorrow and im going to break up with her but i need some advice on REALLY get over her cause im done with this, its an unhealthy relationship and it fucks with my mind.
        """.strip(),
        reference="My [22M] girlfriend [21F] of five years off and on cheated on me by kissing three other dudes. I need help to really break up with her.",
    ),
    dict(
        subreddit="r/Advice",
        title="I got in a fight, but lost. However, I fought to the end.",
        post="""
Okay this is a little silly, and I'm only 16...
So I was just playing football with some people and after the play was done, this dude slapped me. I tried going after him, but a friend held me back.
So about an hour later... still playing... we took a break and I asked to slap-box the guy who slapped me. (It was just a slap, I wouldn't actually fight someone over that.)
He agreed, so we slap-boxed, and I took a lot of hits. I ended up tripping backwards and he got on top and kept slapping me til some guys pulled him off. They said we should be done and we were.
Afterwards, I told him "we're good" and he told me "it's cool, good fight."
        """.strip(),
        reference="I [16] got in a fight after playing football. We agreed to slap-box and I lost, taking a lot of hits until other guys pulled him off. Afterwards we were still cool with each other.".strip(),
    ),
    dict(
        subreddit="r/relationships",
        title="I'm bipolar. My [22F] boyfriend [23M] wanted to go on a cruise but my brother [26M] didn't allow me. My boyfriend is very upset & thinks my brother is controlling my life beyond reasonable.",
        post="""
I'm bipolar. It runs in our family very bad. I'm always under medication and I need to take different medication when I'm close to or on episodes. When I'm taking my medication I'm mostly fine but I still need to be very careful. 2 years ago I petitioned for my brother to become my guardian as I really needed him to take care of me. If I go off of my medication I become unpredictable and I need someone to force me back and limit the damage I can do. Even on medication I sometimes need a kick in the butt. That's why he's my guardian. He can do those things.
So my boyfriend of 6 months and I decided that it will be great to go on a cruise. Now on some level I always knew that my brother will say no but I thought let's try. I told him and predictably he said no. He explained that not having access to me for a week, being on the water for the first time, and being around drinks and alcohol (I shouldn't drink, messes up my treatment) is risky and I shouldn't go. He said it's fine if we want to take a different holiday on land somewhere that's easy to access but he won't allow me to go to a cruise ship.
Ok I was disappointed but I kind of knew he won't allow it. My boyfriend was really frustrated when I told him. He said he really looked forward to this trip and he doesn't want to cancel, he thinks we should still go and my brother can't stop us. He also said my brother is very controlling and he's acting like a parent and he's being an asshole to me for not letting me do what I want to do and he's a abusing his role as guardian. Look I'm disappointed too but I think we can still have fun in a different trip.
I don't know. He's really really upset about this whole thing and I don't know what to do.
        """.strip(),
        reference="I'm bipolar and my brother has guardianship and wouldn't allow me to go on a cruise with my boyfriend of six months.",
    ),
]

for x in tldr_zero_shot_examples:
    response_length = len(encoder.encode(x["reference"]))
    assert (
        response_length < 48 and response_length >= 24
    ), f"{response_length} tokens: {x['reference']}"
tldr_zero_shot_context = """

=====

""".join(
    [
        tldr_zero_shot_format_str.format(**example) + " " + example["reference"]
        for example in tldr_zero_shot_examples
    ]
    + [""]
)
tldr_zero_shot_task = combos(
    tldr_task,
    # bind("response.truncate_token", 628),  # \n\n
    bind("response.truncate_token", 198),  # \n
    bind("query.format_str", tldr_zero_shot_format_str),
    bind("query.padding", tldr_zero_shot_context),
    # max context length (for context stuffing) minus query
    bind("query.length", 2047 - 48),
)

tldr_rm_task = combos(tldr_task, bind("response.length", 128))

tldr_ppo_task = combos(tldr_task, bind("query.dataset", "tldr_3_filtered_queries"))

tldr_zero_shot_cnndm_task = combos(
    tldr_task,
    bind("query.format_str", "{article}\n\nTL;DR:"),
    bind("query.dataset", "cnndm"),
    bind("query.truncate_field", "article"),
    bind("query.truncate_text", "\n"),
)

test_tldr_task = combos(
    bind("query.dataset", "tldr_3_filtered"), bind("query.length", 8), bind("response.length", 4)
)

test_task = combos(
    bind("query.dataset", "test"),
    bind("query.length", 8),
    bind("response.length", 4),
    bind("response.ref_format_str", "{reference}"),
)


def random_teeny_model_spec(**run_params):
    return combos(
        bind("load_path", "https://openaipublic.blob.core.windows.net/summarize-from-feedback/models/random-teeny"),
        *[bind(f"run_params.{k}", v) for (k, v) in run_params.items()],
    )


def stub_model_spec(**run_params):
    return combos(
        bind("device", "cpu"),
        bind("load_path", "https://openaipublic.blob.core.windows.net/summarize-from-feedback/models/random-teeny"),
        *[bind(f"run_params.{k}", v) for (k, v) in run_params.items()],
    )


def load_model_spec(load_path, short_name=None, **run_params):
    return combos(
        bind("load_path", load_path),
        bind("short_name", short_name),
        *[bind(f"run_params.{k}", v) for (k, v) in run_params.items()],
    )


# tldr finetunes
sup4 = partial(load_model_spec, "https://openaipublic.blob.core.windows.net/summarize-from-feedback/models/sup4", short_name="sup4")

rm4 = partial(load_model_spec, "https://openaipublic.blob.core.windows.net/summarize-from-feedback/models/rm4", short_name="rm4")

sup4_ppo_rm4 = partial(
    load_model_spec,
    "https://openaipublic.blob.core.windows.net/summarize-from-feedback/models/sup4_ppo_rm4",
    short_name="sup4_ppo_rm4",
)
