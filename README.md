# Optical Multilayer Simulation and Design Optimization Module

Created by David C. Heson, Jack Liu, and Dr. William M. Robertson over the 2022 MTSU Computational Sciences REU (NSF REU Award #1757493).

Codebase created to explore the optical properties of multilayers and optimize their design to achieve maximum sensibilities, using minimas of reflected light and RIU (Refractive Index Unit) as qualitative measurements. A new measurement called SWT (Shift With Thickness) was also introdued, which is defined as the degree or nanometer shift in the location of a Bloch Surface Wave when the bottom layer increases by 10 nanometers in thickness.

The current functionalities of the program are:
<ul>
<li>Create custom "layer" variables which define depths and indexes of refraction for each layer, with an added functionality to "autobuild" layers which exhibit fixed periodicity.</li>
<li>Graph reflection and transmission coefficients across multilayers with changing angle or wavelength, for specific polarisation mode.</li>
<li>Graph the electric field profile accross a multilayer for a specific angle, wavelength, polarisation, and multilayer.</li>
<li>Calculate RIU and SWT values for specified multilayer arrangements.</li>
<li>Explore by force bruting a set of parameters for the multilayers, finding the best setups in terms of RIU/SWT and reflectivity dips.</li>
<li>Simulate the properties of a multilayer using a dynamic index of refraction, which is wavelength dependent and corresponds to a user-defined function.</li>
<li>More to come!</li>
</ul>

Please refer any questions, comments, or suggestions to dch376@msstate.edu.

To do:
<ul>
<li>Rewrite matrix calculations using Rust.</li>
<li>Explore how to do a gradient descent on parameter regions of interest.</li>
<li>Modularize the written functions into PIP.</li>
<li>Improve help and context messaging.</li>
<li>Improve workflow within the script.</li>
<li>Condense and improve the electric field calculation.</li>
</ul>
