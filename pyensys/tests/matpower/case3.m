function mpc = case9
%CASE9    AC Optimal Power flow data for 3 bus, 1 generator case.
%   MATPOWER

%% MATPOWER Case Format : Version 2
mpc.version = '2';

%%-----  Power Flow Data  -----%%
%% system MVA base
mpc.baseMVA = 100;

%% bus data
%	bus_i	type	Pd	Qd	Gs	Bs	area	Vm	Va	baseKV	zone	Vmax	Vmin
mpc.bus = [
	1		2		10	5	0	0	1		1	0	345		1		1.1		0.9;
	2		3		0	0	0	0	1		1	0	345		1		1.1		0.9;
	3		2		15	10	0	0	1		1	0	345		1		1.1		0.9;
];

%% generator data
%	bus	Pg	Qg		Qmax	Qmin	Vg		mBase	status	Pmax	Pmin	Pc1	Pc2	Qc1min	Qc1max	Qc2min	Qc2max	ramp_agc	ramp_10	ramp_30	ramp_q	apf
mpc.gen = [
	1	0	0		0		0		1.04	100		1		0		0		0	0	0	0	0	0	0	0	0	0	0;
	2	0	0		30		0		1.025	100		1		40		0		0	0	0	0	0	0	0	0	0	0	0;
	3	0	0		0		0		1.025	100		1		0		0		0	0	0	0	0	0	0	0	0	0	0;
];

%% branch data
%	fbus	tbus	r		x		b	rateA	rateB	rateC	ratio	angle	status	angmin	angmax
mpc.branch = [
	1		2		0.1		0.5		0.02	12	0	0	0	0	1	-360	360;
	1		2		0.1		0.5		0.02	12	0	0	0	0	0	-360	360;
	2		3		0.1		0.33	0.01	19	0	0	0	0	1	-360	360;
	2		3		0.1		0.33	0.01	19	0	0	0	0	0	-360	360;
];

%%-----  OPF Data  -----%%
%% generator cost data
%	1	startup	shutdown	n	x1	y1	...	xn	yn
%	2	startup	shutdown	n	c(n-1)	...	c0
mpc.gencost = [
	2	0	0	2	4	0;
	2	0	0	2	2	0;
	2	0	0	2	4	0;
];
