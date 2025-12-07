# Contributing to MGS

Guidelines for contributing to the Model Graph Simulator project.

## Development Workflow

### Git Commit Tags

When presenting work with MGS/NTS components to a client, create a git tag and record it in the [Box note](https://ibm.ent.box.com/notes/231444066519).

See instructions inside the Box note for:
- Creating tags
- Naming conventions
- Documentation requirements

### Code Organization

- **Framework code**: `gsl/framework/`
- **Model definitions**: `models/*.mdf`
- **Examples**: `examples/`
- **Documentation**: `docs/`

## ZenHub Project Management

Based on [ZenHub's documentation](https://www.zenhub.com/github-project-management.pdf) and [IBM's ZenHub documentation](https://pages.github.ibm.com/the-playbook/zenhub/).

### Overview

ZenHub extends GitHub's issues with Agile project management features.

**Agile Concepts → ZenHub/GitHub:**
- Sprint/Iteration → GitHub Milestone
- User Stories → GitHub Issues
- Epics → Epics
- Product backlog → Open issues without a Milestone
- Sprint backlog → Issues with a Milestone
- Tasks → GitHub Issues (with sub-tasks as Markdown checklists)
- Projects → Epics

**Priority:** Issues should be arranged by priority (most pressing at top) within Epics.

### Creating Issues

1. Select the "Issues" tab
2. Click "New issue"
3. Use the template provided:
```
   As a <user type>, I want to <task>, so that <goal>, whereas currently <this does not exist>.
```
   
   **Examples:**
   - "As a CHDI researcher, I want to understand how MSN firing patterns change during HD progression, whereas currently only WT is characterized."
   - "As a MGS user, I want to store the current state of the RNG to disk so that I can restart a simulation with this state, whereas currently this is not possible."

4. Add sub-tasks with Markdown
5. Assign team member(s) (optional)
6. Add labels
7. Add milestone (optional)
8. Estimate complexity in hours
9. Place inside an epic (optional)
10. Click "Submit new issue"

### Creating Epics

Best to create Epics after adding constituent Issues.

1. Select "Issues" tab
2. Click "New issue"
3. Give the issue a title, remove the description
4. Click "Create an epic"
5. Select Issues that are part of the Epic
6. Click "Create Epic"

### Estimating Complexity

Estimates represent relative complexity, not absolute time/effort.

See [ZenHub's complexity guide](https://www.zenhub.com/github-project-management.pdf) for details.

### Pipelines

- **New Issues**: New issues land here automatically. Triage immediately.
- **Icebox**: Not a current focus, but will address eventually.
- **Backlog**: Current focus
  - Without milestone = product backlog
  - With milestone = sprint backlog
- **In Progress**: Currently being worked on
- **Review/QA**: Awaiting review or QA (optional)
- **Done**: Finished, no further work required
  - `Target: Science` → Move here when client/community is notified (meeting, report, white paper) AND code committed/pushed
  - `Target: MGS/NTS` → Move here when code committed/pushed AND pull request issued with code review scheduled
  - Before PR: Consider [merging or rebasing](https://www.atlassian.com/git/tutorials/merging-vs-rebasing) master into branch
- **Closed**: Completed work
  - `Target: Science` → Move here after peer-reviewed publication AND code merged to master
  - `Target: MGS/NTS` → Move here after pull request accepted

### Labels

Labels are grouped: Client, Priority, Status, Target, Type

**Status Labels:**
- **Status: Coding** - Coding in progress
- **Status: Design** - Still being designed
- **Status: Discussion** - Under discussion
- **Status: Requirements** - Requirements analysis underway
- **Status: Results Analysis** - Analyzing scientific results
- **Status: Simulating** - Running simulations

**Target Labels:**
- **Target: MGS** - Targeted at MGS framework
- **Target: NTS** - Targeted at NTS (Neural Tissue Simulator)
- **Target: Science** - Scientific research of general interest

**Type Labels:**
- **Type: Bug** - Fix required for progress
- **Type: Enhancement** - Extension/modification to facilitate progress
- **Type: Research** - Background work, testing, preliminary results
- **Type: Maintenance** - Work for future code/model reuse

## Contributing Guidelines

1. **Fork the repository** (for external contributors)
2. **Create a feature branch** from `dev`
3. **Make your changes** following code style guidelines
4. **Test thoroughly** before submitting
5. **Submit a pull request** to `dev` branch
6. **Respond to review feedback**

## Code Style

- Follow existing code conventions
- Comment complex algorithms
- Use descriptive variable names
- Keep functions focused and modular

## Testing

Before submitting:
- Ensure code compiles without warnings
- Run existing test suite
- Add tests for new features
- Verify GPU code on actual hardware (if applicable)

## Documentation

- Update README.md for user-facing changes
- Update BUILD.md for build process changes
- Add comments for complex code
- Update docs/ for API changes

## Useful Resources

- [GitHub Markdown Guide](https://guides.github.com/features/mastering-markdown/)
- [Git Merging vs Rebasing](https://www.atlassian.com/git/tutorials/merging-vs-rebasing)
- [ZenHub Documentation](https://www.zenhub.com/github-project-management.pdf)

## Questions?

See:
- [README.md](README.md) - Project overview
- [BUILD.md](BUILD.md) - Build instructions
- [docs/](docs/) - Technical documentation

Or open an issue for discussion!
