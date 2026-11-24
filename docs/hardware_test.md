# Manual hardware test checklist

Run this on the real machine before trusting BasiCAM CNC with a job. Keep the physical E-stop in reach for every step. Work through it with the spindle off and no tool or stock unless a step says otherwise. Tick each box and note the date, machine and controller firmware.

Machine: ______________ Controller / firmware: ______________ Date: __________ Tester: __________

## 1. Connection and identification

- [ ] **Machine > Ports...** lists the controller's port (with its USB chip) and adds it within 2 s of plugging it in.
- [ ] **Connect** waits for the start-up message, then shows the firmware and version (`$I`) and the header reads "Connected - not homed".
- [ ] `status` in the console prints only `;` comment lines and sends nothing that moves the machine.
- [ ] The preflight window opens once the position is verified (Home, or Verify position); normal jogs stay disabled until every axis is ticked.
- [ ] After a soft reset or alarm, the preflight window opens again once the position is verified. After a soft-limit stop it does not.

## 2. Limits and keep-out

- [ ] **Soft limits...**: jog each axis to its endstop, press **Record**; the stored values match the controller's machine position.
- [ ] The default margin is 20 mm. Below 5 mm, or **Released**, is refused unless the gate is on.
- [ ] Z: with the longest tool mounted (stick-out = overall length - 12 mm), the lowest Z is recorded; a shorter tool may go lower by the difference.
- [ ] A jog that would put an axis past its soft limit is refused with an alarm; one into the hard zone is refused even with the gate on.
- [ ] An axis can be jogged to within 1 mm of its soft limit (and back there after moving away); the tool outline is not counted against the travel.
- [ ] With a clamp set up as a fixture, a jog that brings the tool within 5 mm of it (cutter included) is refused; a feed move into stock deeper than the tool's cutting length is refused.
- [ ] Add a keep-out zone (machine setup or a fixture marked **Keep out**): jogs into it are refused; jogs over it with clearance are allowed.

## 3. Motion

- [ ] Homing (**Home ($H)**) completes and the header reads "Connected - ready".
- [ ] Each jog-pad arrow moves the right axis in the right direction by that axis's step, at the XY or Z feed (see section 14).
- [ ] Keyboard control, **Virtual** on: arrows move only the ghost and the DRO; nothing is sent (check the console).
- [ ] Keyboard control, Virtual off: X / + / 5 / Enter shows a ghost; the second Enter moves 5 mm; Esc at each step backs out.
- [ ] **End** feed-holds; **Resume** continues; **KILL** (pad, status bar or any machine window) stops the spindle and retracts Z without hitting the top stop.

## 4. Speed multiplier

- [ ] The first change above 100% asks once per session.
- [ ] 150%: the controller reports feed override 150% (and spindle 150% unless the spindle speed is manual).
- [ ] Above 200% with the gate off: capped at 200%. With the gate on: streamed F words are scaled and never exceed $110-$112.

## 5. Programs (air cut first, Z raised well above the table)

- [ ] Generate a simple profile from a fully defined, anchored sketch; the program starts with units, G90 and the setup's work offset.
- [ ] Inches display: the program uses G20 with inch values, including the final retract.
- [ ] Advanced G-code (gate on): **Run program** shows the disclaimer, streams one line per `ok`, and logs every line.
- [ ] While running, the editor is read-only. After stopping (spindle off), it becomes editable.
- [ ] Stop part-way, close and reopen the project: a resume is suggested from the last acknowledged line.
- [ ] **Resume code here** on a later line: asks for the tool and home location; the preamble raises to safe Z, moves to XY, then plunges.
- [ ] A program that enters a keep-out zone refuses to start.

## 6. Faults

- [ ] Trigger a limit switch by hand during a slow jog: the header shows "ALARM 1"; the terminal shows a timestamped description.
- [ ] Send an invalid line (`G99`): a timestamped "Error 20" appears in the terminal.

## 7. First real cut

- [ ] Soft stock (wax, foam or MDF), a known tool, conservative feeds; watch the whole program with a hand on the E-stop.
- [ ] Measure the part and compare it with the sketch dimensions.

## 8. Laser, plasma, waterjet and wire EDM (air run first, beam / arc / jet disabled at the source)

- [ ] **Laser test fire** is refused until Settings > Advanced > Allow laser Test fire is on; with it on, the beam fires at 1% for 0.1 s with safety glasses on and the enclosure closed.
- [ ] A cut program: the console shows the beam command (M3 / M4 S...) only after the pierce position is reached, and M5 before every rapid. No G0 line ever appears between a beam-on and the following M5.
- [ ] Plasma: pierce height, pierce delay (G4) and cut height match the cut chart; torch-height control switches as configured.
- [ ] Waterjet / EDM (export only): the exported file runs in the machine's own software in a dry run.
- [ ] Tabs / micro-joints hold the parts; lead-ins never touch a neighbouring part; inner contours are cut before the outline.
- [ ] Flash cutting (only if the laser has fast beam gating): a hole grid runs in rows without stopping at each hole.

## 9. Rotary axis (4th axis or rotary laser)

- [ ] **Rotary axis setup** matches the device (headstock diameter, throw, mount, tailstock); the view shows its chuck and jaws in red.
- [ ] `[` and `]` turn the rotary axis by the step; a turn beyond a non-continuous limit is refused.
- [ ] A wrapped square on known-diameter stock measures the sketch size along the surface.
- [ ] Indexed setups at 0 / 90 / 180 / 270 degrees index before their passes at safe Z.

## 10. Lathe

- [ ] The controller is in lathe mode (grblHAL `$32=2`) and X reads the diameter (G7).
- [ ] A jog of the tool towards the jaws is refused; facing the free end is allowed.
- [ ] Threading is offered only when the controller supports G76.
- [ ] A manual-speed headstock pauses with the RPM prompt.
- [ ] **Open chuck** is refused while the spindle turns.
- [ ] Bar feed: a production run of 3 parts parts off, stops the spindle, opens the chuck, advances the bar to the stop, closes the chuck and continues; the job report shows 3 parts and the bar used.

## 11. Connection: firmware and baud rate (each controller you use)

Machine: ______________ Firmware: GRBL / grblHAL / FluidNC ______ Baud set in the firmware: ______

- [ ] At the firmware's baud rate, **Connect** identifies the firmware within the boot wait (3 s by default) on a board that resets when the port opens (Arduino-based GRBL).
- [ ] At a wrong baud rate, the panel says "wrong baud rate?" (or "No reply") and lists the likely causes; nothing is enabled.
- [ ] **Baud rate: Auto** finds the firmware's rate and stores it on the machine.
- [ ] With the port open in another program, the error says so and nothing hangs.
- [ ] **Test connection** shows firmware, version, options, axes, state, inputs and the round-trip time, and moves nothing.
- [ ] Unplugging the cable while connected shows "Controller not responding" within about 5 s; a running program is feed-held.
- [ ] grblHAL and FluidNC: the header shows the right firmware name; Home, jog and the console behave as with GRBL.

## 12. From connected to moving (step by step; write down what the terminal shows)

Follow the machine panel's **Ready checklist** from the top. At each step note the checklist item highlighted and the last terminal line.

1. Connected: ______________________________________________
2. Firmware identified: ____________________________________
3. Not in alarm (Unlock or Home, as offered): _______________
4. Position verified (Home, or Verify position for a machine without switches): ______
5. Preflight done (preflight window, each axis moved both ways and ticked): ______
6. Ready: one pad jog of 1 mm on each axis moves exactly 1 mm: ______

- [ ] Every disabled button shows why in its tool tip, and every refused click writes the reason in the terminal and status bar.
- [ ] A controller that starts in alarm with homing enabled offers **Home** (not only Unlock), and homing clears it.
- [ ] Machine without homing switches: **Verify position** asks where the tool is; the DRO reads "position set by you"; a jog beyond the travel is refused.

## 13. Alarms and the limit-switch test

- [ ] The inputs line in the machine panel shows each limit as you press it by hand.
- [ ] **Limit-switch test**: press and release every limit switch; each row turns ON then off and shows "works".
- [ ] If every limit reads ON with nothing pressed: the panel explains it (inverted pins, normally-closed switches, floating inputs, noise) and shows `$5`, `$21`, `$22`. **Invert limit pins ($5)** sends the change, then the test starts again: press every switch again and confirm each one works. FluidNC: the test says to change `config.yaml` instead.
- [ ] `$21=0`, `$20=0` and `$5=...` typed in the console are refused.
- [ ] An alarm shows its meaning for the identified firmware and what to do next.

## 14. Jog pad: directions and steps

- [ ] With the machine's axes as labelled, X+ / X− / Y+ / Y− move the way the arrows point; if not, **Flip X arrows** / **Flip Y arrows** fixes it and is remembered for this machine.
- [ ] The corner arrows move X and Y together in one move.
- [ ] Each step (0.01, 0.1, 1, 10, 100 mm, or the inch steps) moves exactly that far; measure 10 mm and 100 mm with a rule or indicator.
- [ ] **Custom...** `0.05in` moves 1.27 mm and stays in the list after restarting BasiCAM.
- [ ] **Link X/Y** on: changing the X step changes the Y step; off: they are separate. Z always has its own step.
- [ ] **Hold to move**: the axis moves while the arrow is held and stops at once on release.
- [ ] **XY 0** asks first, raises Z to a safe height, then goes to work X0 Y0.
- [ ] Rotary machine: the A (or B / C) row turns by its step in degrees.
- [ ] Keyboard arrows use the pad's steps (Shift x10, Ctrl /10).

## 15. Preflight jogs

- [ ] Before the position is verified the preflight window offers only Home and Unlock (and KILL).
- [ ] After verifying, the pad in the preflight window jogs each axis; a jog into a keep-out or past a limit is refused and nothing moves.
- [ ] An axis's tick is offered only after it has moved both ways; normal jogs work only after every axis is ticked.
- [ ] The console and DRO stay usable while the preflight window is open.
