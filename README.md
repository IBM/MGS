./make_nts  [print out help instruction]
./make_mgs  [print out help instruction]
./make_both  [print out help instruction]


GSL:
./build_gsl -h     [print out help instruction]
  ./build_gsl --rebuild --release -d 4   is good for release
  ./build_gsl --rebuild -d 4 is good for debugging


NTI:
 make debug=yes    debug
 make              release


External library:
  bison 
  flex
  lgmp



# ZenHub
The following is based on the reading of [ZenHub's documentation](https://www.zenhub.com/github-project-management.pdf) 
and other tutorial and help guides.

## ZenHub overview
ZenHub extends GitHub's issues tool with extra features for Agile project management. 

Agile has different concepts and they link to ZenHub/GitHub as follows:
* Sprint/Iteration - GitHub Milestone
* User Stories - GitHub Issues
* Epics - Epics
* Product backlog - Open issues without a Milestone
* Sprint backlong - Issues with a Milestone
* Further, you can think of GitHub Issues as tasks, and can include sub-tasks as Markdown Checklistsinside an Issue, and think
of Epics as projects.

The core of ZenHub are GitHub issues, and they should be arranged by priority, with the most pressing issues at the top 
(within Epics only).

When you make a new Issue a template is available to keep consistency across Issues, see FAQ below.

## Useful links
* [Description of GitHub Markdown](https://guides.github.com/features/mastering-markdown/)

## FAQ

### How do I add a new Issue?
1. Select the "Issues" tab
2. Click "New issue"
3. A template is provided for you. The first line after the comment can be used as the title and you can 
replace `<user type>`, `<task>`, `<goal>` and `<this does not exist>` appropriately. For example: 
"As a CHDI researcher, I want to understand how MSN firing patterns change during HD progression, 
whereas currently only WT is characterized." or "As a MGS user, I want to store the current state 
of the RNG to disk so that I can restart a simulation with this state, whereas currently this is not possible."
4. Add sub-tasks with Mardown and/or more description about the issue.
5. Assign a/some team member(s) (optional).
6. Add labels.
7. Add a milestone (optional).
8. Estimate the time to complete in hours (see "How should I estimate complexity?" below).
9. Place it inside an epic (optional).
10. Click "Submit new issue"

### How do I add a new Epic?
It is best to add an Epic after already adding some Issues that will be part of the Epic.
1. Select the "Issues" tab
2. Click "New issue"
3. Give the issue a title and remove the description under "Write"
4. Click "Create an epic"
5. Select Issues that are part of the Epic.
6. Click "Create Epic"

## How should I estimate complexity?
When estimating, the units aren't time or effort etc. rather the number represents how complex the Issues are in
*relation* to each other.
See [this document](https://www.zenhub.com/github-project-management.pdf) for a more in depth discussion.

### What are the pipelines?
**New Issues**: New Issues land here automatically. They should be dragged to another pipeline as soon as possible.
**Icebox**: Issues that *are not* a current focus, but you will act on them at some point.
**Backlog**: Issues that *are* a current focus.  If they don’t have a GitHub milestone, consider them part of your
*product backlog*. Once you add a Milestone, they’re part of your *sprint backlog*.
**In Progress**: What's being worked on right now.
**Review/QA**: For Issues that are on hold as need review or QA. [Optional]
**Done**: Finished! No further work required as part of this task. If it is `Target: Science` keep it here until the client 
or community is aware through a report/publication. If it is either `Target: MGS` or `Target: NTS` keep it here for a while
for all users to be aware of its completion.
**Closed**: View your completed work. Drag issues here to close them for all users; dragging them out will re-open them. If it 
is `Target: Science` move it here after the client or community is aware through a report/publication. If it is either 
`Target: MGS` or `Target: NTS` move it here after all users are aware of its completion.

### What are the labels?
The labels are split in to 5 groups, Client, Priority, Status, Target and Type. Descriptions of some labels follow.

**Status: Coding**: Coding of the task is underway.
**Status: Design**: Issue is still being designed.
**Status: Discussion**: The issue is being discussed.
**Status: Requirements**: Requirements analysis is underway.
**Status: Results Analysis**: Scientific results from some simulations are being analysed.
**Status: Simulating:**: Simulations are being run to address the task.

**Target: MGS**: The task is targeted at MGS.
**Target: NTS**: The task is targeted at NTS.
**Target: Science**: The task is not specifically targeted at MGS or NTS but rather is a general scientific task.

**Type: Research**: When the task is neither a bug, enhancement or maintenance of the code base but rather is a general 
research task.



