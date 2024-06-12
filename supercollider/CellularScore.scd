s.boot;

NetAddr.localAddr

(
SynthDef(\bpfsaw, {
	arg atk=1, sus=0, rel=0, c1=1, c2=(-4),
	freq=500, cf=700, rq=0.2, amp=1, out=0;
	var sig, env;
	env = EnvGen.kr(Env([0,1,1,0],[atk,sus,rel],[c1,0,c2]),doneAction:2);
	sig = Saw.ar(freq);
	sig = BPF.ar(sig,cf,rq);
	sig = sig * env * amp;
	Out.ar(out, sig);
 }).add
)

(
OSCdef('PlayS', {|msg|
	~var1 = msg[1];
	~var2 = msg[2] * 0.5;
	Pbind(\instrument, \bpfsaw, \degree, Pseq([~var1], 1), \dur, 0.1, \amp, ~var2).play;
}, "/PlayS");
)
//

(
OSCdef('endS',{
	Pbind(
		\note, Pseq([-1,0,6.1], 1),
		\dur, 0.5,
		\amp, 0.6,
	).play;
}, "endS");
)

s.quit;