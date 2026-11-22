# Manual hardware test checklist

Run this on the real machine before trusting BasiCAM CNC with a job. Keep the physical E-stop in reach for every step. Work through it with the spindle off and no tool or stock unless a step says otherwise. Tick each box and note the date, machine and controller firmware.

Machine: ______________ Controller / firmware: ______________ Date: __________ Tester: __________

## 1. Connection and identification

- [ ] The serial port appears after **Refresh**; **Connect** shows the GRBL version (`$I`) and the header reads "Connected - not homed".
- [ ] `status` in the console prints only `;` comment lines and sends nothing that moves the machine.
- [ ] The preflight dialog opens after connecting; the jog buttons stay disabled until every axis is ticked.
- [ ] After a soft reset or alarm, the preflight dialog opens again. After a soft-limit stop it does not.

## 2. Limits and keep-out

- [ ] **Soft limits...**: jog each axis to its endstop, press **Record**; the stored values match the controller's machine position.
- [ ] The default margin is 20 mm. Below 5 mm, or **Released**, is refused unless the gate is on.
- [ ] Z: with the longest tool mounted (stick-out = overall length - 12 mm), the lowest Z is recorded; a shorter tool may go lower by the difference.
- [ ] A jog that would put the tool envelope past a soft limit is refused with an alarm; one into the hard zone is refused even with the gate on.
- [ ] Add a keep-out zone (machine setup or a fixture marked **Keep out**): jogs into it are refused; jogs over it with clearance are allowed.

## 3. Motion

- [ ] Homing (**Home ($H)**) completes and the header reads "Connected - ready".
- [ ] Each jog button moves the right axis in the right direction by the set step, at the set feed.
- [ ] Keyboard control, **Virtual** on: arrows move only the ghost and the DRO; nothing is sent (check the console).
- [ ] Keyboard control, Virtual off: X / + / 5 / Enter shows a ghost; the second Enter moves 5 mm; Esc at each step backs out.
- [ ] **End** feed-holds; **Resume** continues; the kill button stops the spindle and retracts Z without hitting the top stop.

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
