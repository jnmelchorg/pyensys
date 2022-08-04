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
    1.0000    2.0000   10.0000    5.0000         0         0    1.0000    1.0000         0  345.0000    1.0000    1.1000    0.9000
    2.0000    3.0000         0         0         0         0    1.0000    1.0000         0  345.0000    1.0000    1.0000    1.0000
    3.0000    2.0000   15.0000   10.0000         0         0    1.0000    1.0000         0  345.0000    1.0000    1.1000    0.9000
];

%% generator data
%	bus	Pg	Qg		Qmax	Qmin	Vg		mBase	status	Pmax	Pmin	Pc1	Pc2	Qc1min	Qc1max	Qc2min	Qc2max	ramp_agc	ramp_10	ramp_30	ramp_q	apf
mpc.gen = [
    2.0000         0         0   30.0000         0    1.0250  100.0000    1.0000   40.0000         0         0         0         0         0         0         0         0         0         0         0         0
];

%% branch data
%	fbus	tbus	r		x		b	rateA	rateB	rateC	ratio	angle	status	angmin	angmax
mpc.branch = [
    1.0000    2.0000    0.1000    0.5000    0.0200   12.0000         0         0         0         0    1.0000 -360.0000  360.0000
    1.0000    2.0000    0.1000    0.5000    0.0200   12.0000         0         0         0         0         0 -360.0000  360.0000
    2.0000    3.0000    0.1000    0.3300    0.0100   19.0000         0         0         0         0    1.0000 -360.0000  360.0000
    2.0000    3.0000    0.1000    0.3300    0.0100   19.0000         0         0         0         0         0 -360.0000  360.0000
];

%%-----  OPF Data  -----%%
%% generator cost data
%	1	startup	shutdown	n	x1	y1	...	xn	yn
%	2	startup	shutdown	n	c(n-1)	...	c0
mpc.gencost = [
	2     0     0     2     4     0
];