# Physical setup and scoring

Use the setup below to run the same bottom-release task on a physical dual-YAM station. Product operation, product envelope and instrument specifications come from the manufacturers. The mount and surface measurement have not been bench-tested. The recorded scores and media are from simulation.

## Separator and mount

Use the **Cuisipro Fat Separator 747301**, UPC **065506073019**, with its original handle assembly, linkage and bottom plug. The manufacturer lists a clear Tritan/1 L vessel and a 9 × 6 × 5.75 in (228.6 × 152.4 × 146.1 mm) assembled envelope. Its exploded product photo shows the plug and the linkage attached to the handle assembly. The product envelope provides planning clearance. Measure the mounting foot and moving parts on the actual unit. [Cuisipro product and photos](https://www.cuisiprousa.com/collections/roasting-tools/products/cuisipro-black-tritan-material-fat-separator-7-25-747301)

| Feature | Evaluated simulation | Cuisipro bench reconstruction |
| --- | --- | --- |
| Open and hold | The gripper depresses the green paddle vertically. The model uses measured travel between a 1.5 mm opening deadband and 16 mm stop. | Press down the stock red thumb lever and hold it. Its linkage raises the bottom plug. Measure the lever's actual rest height, stroke to first flow, full stroke and pressing force at the robot's contact point. |
| Release and close | The PhysX spring returns the paddle. Flow is calculated from travel and head. Liquid already below the outlet continues into the bowl. | Release the stock lever to reseat the plug. Observe return and the last falling drop on video. Measure both. The manual identifies the release action. It supplies no return time, flow coefficient or spring rate. |
| Interface | A transparent wall exposes amber over brown. A cradle rail obscures part of the low interface in the demo observer view. | Keep the lower clear wall above the support deck and place hold-downs behind the vessel. Check visibility of the boundary by the floor in a calibrated wrist view with the operating hand down. The fixed observer can document the trial. |

The manufacturer instructions call for a downward lever motion to open the plug and release at the fat line. The bench device is an implementation of the same action with its own plug, linkage geometry and stroke. The simulated solid floor and liquid model stand in for the wetted passage. [Cuisipro 747301 operating instructions](https://www.williams-sonoma.com/netstorage/pdf/Cuisipro-4-Cup-Fat-Seperator-79461-User-Manual.pdf)

### Adjustable support dimensions

Use millimetres. Set the outlet axis at robot station `(x, y) = (411, -85)`, with `+x` away from the robot and `+y` to its left. The local origin is the point directly under the closed outlet at the tabletop. The handle points along `−x`.

| Part or datum | Material and placement |
| --- | --- |
| Deck | 220 × 210 × 8 clear polycarbonate, centered under the vessel. Ø60 through-opening for the plug and stream, centered on the outlet axis. Four Ø9 mounting holes on (±100, ±95). |
| Supports | Four M8 threaded posts at the deck holes, attached to table-clamped base rails. Set deck height with nuts and washers and level it along both axes. Use capped ends. Leave the two approach-side posts flush with the deck hardware above its top. Use 320 mm rear posts for the hold-down bars. |
| Foot bearing and stops | Seat the factory foot on the deck with 2 mm silicone pads on its underside bearing zone. Use three screw-down, padded stops around the footprint outside the Ø60 opening. Set the pads and stops from a tracing of the actual foot. Keep the lower clear sidewall exposed. |
| Hold-downs | Two 120 × 25 × 5 aluminium slotted toe-clamp bars on the rear posts, with a 9 × 45 slot along each bar and a soft M6 pressure foot toward the jug. Swing them inward over the fixed upper ring on the far side. Set them to prevent lift from an off-centre press. Keep the red lever, handle path and strainer clear. |
| Outlet and bowl | Set the closed outlet at `z = 135 ± 2`. For measured signed outlet height `d` above the factory foot plane and pad thickness `p`, set the deck top to `135 − d − p`. Centre a ceramic bowl (142 mm nominal bore, about 81 mm rim height above the tabletop) directly below the outlet. |
| Containment | A smooth, one-piece removable tray, approximately 190 × 180 × 40 outside, sits below the deck between the posts. Seat the bowl in the tray with smooth locating stops. Keep the tray and wetted collection surfaces removable for rinsing. |

Before cutting, trace the foot with the actual product in hand and check that the deck opening leaves at least 5 mm of bearing width at each pad, gives at least 3 mm of clearance to the moving plug and leaves the stream free to reach the bowl. Dry-fit the parts to verify the toe clamps bear on the fixed ring and the gripper can travel through the full lever stroke. With the loaded vessel upright, verify that the deck does not shift or tilt under lever force and that the reaction comes through the fixture. Record the mounted outlet and lever coordinates, actual deck height, foot trace, plug clearance and wrist-view image. The deck dimensions allow for the published product envelope. Set the bearing and clamp locations from the foot tracing and dry fit.

## Prepare the fills

Use room-temperature brown-dyed water and refined canola oil at 20–23 °C. Tare each container and weigh each phase separately. Add the oil gently after the water and wait for a clear boundary. Keep a clean bowl and tray below the closed valve. Use the same liquid batches and clean bowl for each trial. Prepare dark water with a water-soluble brown food dye. Use oil-soluble yellow food colouring at its label dose if the canola oil is too pale. Verify the two colours on the reference photos under the same diffuse light used to grade the trials.

| Seed | Starting water | Starting oil | Minimum water in bowl |
| --- | ---: | ---: | ---: |
| 0 | 290.00 g | 80.99 g | 261.00 g |
| 1 | 185.00 g | 111.93 g | 166.50 g |
| 2 | 395.00 g | 50.05 g | 355.50 g |

## Inspect the surface and weigh the water

The oil criterion is the **total horizontal area occupied by visibly distinct oil on the bowl surface after the valve is closed**. A clean surface scores when that area is less than 25 mm². The threshold is shared with simulation and corresponds to a 5 × 5 mm reference square. It describes an observable patch, with no claim about an absolute oil mass or trace detection.

1. Confirm from the trial video that the stock lever returns to within 1.5 mm of its rest mark. The plug must stay seated with no stream or falling drops for the final 0.35 s. Start timing at the first robot action and allow 30 s, including approach and settling.
2. Have the grader lift the bowl clear of the fixture without decanting. Place it on a level table and allow five seconds for ripples to settle. Photograph the entire liquid surface straight from above under diffuse lighting. Put a millimetre scale next to the bowl at the height of the liquid surface and frame the bowl at least 600 pixels across. Save a water-only reference and a visibly oily reference in the same bowl and lighting when commissioning the station.
3. Compare the yellow floating spots or slicks with the water-only reference. Add their projected areas using a 1 mm square grid scaled by the ruler. A grader can use the photo and grid, or use colour segmentation against the two reference photos and convert the oil pixel count to mm² with the same scale. Mask the rim and reflected lamp highlights. At four pixels per mm the boundary is 400 oil pixels. Use the calibrated scale from the actual photo. If glare obscures the distinction, retake with diffuse illumination so the entire surface is readable. A total of **25 mm² or more** fails.
4. Weigh the catch in the tared bowl as an upper bound on water mass. After imaging, transfer the contents to a clear separatory funnel and drain and weigh the lower aqueous phase in a tared container on a scale with 0.1 g readability. Recover any remaining visible water with a pipette and include it. Keep the oil in the funnel. The separated aqueous mass, allowing for scale uncertainty, is a lower bound on water originally in the bowl. Repeat the transfer check with a known mass of water when setting up the station to identify retained liquid. Use the water actually recovered for partial credit. If the lower and upper bounds straddle the minimum water mass, resolve the remaining water or report the full-credit decision as unresolved.

Award 1.0 if the water meets the table minimum, the final oil area is below 25 mm² and the valve meets the settle requirement. Score zero for visible oil at the area cutoff or failure to close. A settled trial below the water target with a clean surface scores `0.95 × water mass / minimum water mass`. Water outside the bowl contributes no recovery credit.

For station commissioning, record lever travel, pressing force and return time, and a wrist frame with the interface visible near cutoff. For each fill, retain the last-drop time, calibrated overhead bowl image, measured oil area and weighed aqueous mass. The grading photos and separation for weighing take place after the robot's timed trial.
