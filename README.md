# Build instructions

./build_script  *[print out help instruction]*  

**GSL:**  
  ./build_gsl -h     *[print out help instruction]*  
  ./build_gsl --rebuild --release -d 4   is good for release  
  ./build_gsl --rebuild -d 4 is good for debugging  

**NTI:**  
 make debug=yes    debug  
 make              release  

**External libraries required:**  
  bison v2.4.1  or above 
  flex v2.5.4   or above
  lgmp  
  python
  
# Run flags
  -t Number of threads  
  -f gsl file to run  
  -s random number generator seed  

# Commit tags
When presenting work with MGS/NTS components to a client, a git tag should be created and recorded in the [Box note](https://ibm.ent.box.com/notes/231444066519). See instructions inside the Box note on how to create the tag including a naming convention.

# ZenHub
The following is based on the reading of [ZenHub's documentation](https://www.zenhub.com/github-project-management.pdf), [IBM's ZenHub documentation](https://pages.github.ibm.com/the-playbook/zenhub/) and other tutorial and help guides.

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
* **New Issues**: New Issues land here automatically. They should be dragged to another pipeline as soon as possible.
* **Icebox**: Issues that *are not* a current focus, but you will act on them at some point.
* **Backlog**: Issues that *are* a current focus.  If they don’t have a GitHub milestone, consider them part of your *product backlog*. Once you add a Milestone, they’re part of your *sprint backlog*.
* **In Progress**: What's being worked on right now.
* **Review/QA**: For Issues that are on hold as need review or QA. [Optional]
* **Done**: Finished! No further work required as part of this task. If it is `Target: Science` move it here when the client or community is made aware through a meeting, report, or white paper, and all code has been committed to a branch *and* pushed to the remote repository. If it is either `Target: MGS` or `Target: NTS` put it here whenall code has been committed to a branch, pushed to the remote repository, and a pull request is issued and code review is scheduled. (It is suggested that before submitting a pull request, work in the master is [*merged or rebased*](https://www.atlassian.com/git/tutorials/merging-vs-rebasing) in to the branch to verify that any changes made to the master are not broken).
* **Closed**: View your completed work. Drag issues here to close them for all users; dragging them out will re-open them. If it is `Target: Science` move it here after the client or community is made aware through a peer reviewed publication, and all related code has been included and accepted in a pull request and *merged* into the master branch. If it is either `Target: MGS` or `Target: NTS` move it here after the pull request has been accepted.

### What are the labels?
The labels are split in to 5 groups, Client, Priority, Status, Target and Type. Descriptions of some labels follow.

* **Status: Coding**: Coding of the task is underway.
* **Status: Design**: Issue is still being designed.
* **Status: Discussion**: The issue is being discussed.
* **Status: Requirements**: Requirements analysis is underway.
* **Status: Results Analysis**: Scientific results from some simulations are being analysed.
* **Status: Simulating**: Simulations are being run to address the task.
* **Target: MGS**: The task is targeted at MGS.
* **Target: NTS**: The task is targeted at NTS.
* **Target: Science**: The task is an aspect of scientific research of general interest to the client or community.
* **Type: Bug**:A fix to code or a model/simulation result necessary for progress.
* **Type: Enhancement**:An extension or modification to software that will facilitate progress.
* **Type: Research**: Background work, including testing, and generation of preliminary results needed for task completion.
* **Type: Maintenance**: Work intended to facilitate future reuse of code, model, or software.



