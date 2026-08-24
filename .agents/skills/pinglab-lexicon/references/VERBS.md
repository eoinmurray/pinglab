# Verbs

## `ScopeEncode`

`scope encode RULE` requests a read-only specification for the smallest
coherent change that brings `RULE` into the Pinglab writing system. Recognize
`scope encode` regardless of case and allow `$scope`, `$encode`, or both. The
command does not authorize editing; a later Lexicon `go` mode supplies that
authority.

To encode a rule:

1. State the observable writing decision that the rule must change.
2. Check the live skill for an existing rule with the same effect. Return a
   no-op when it already exists, or refine the existing rule instead of adding
   a duplicate.
3. Reject a one-off edit, vague preference, anecdote, or rule that would weaken
   scientific accuracy, epistemic boundaries, user intent, or a noun contract.
4. Select the narrowest owner: one noun, one noun family, the shared
   human-facing writing contract, or recognition metadata when invocation must
   change.
5. Integrate the rule into the owner's existing instructions. State its scope,
   precedence, and material exclusions without growing an exception list.
6. Specify one positive case, one boundary case, and one conflicting case that
   the implementation must handle correctly.
7. Validate the changed skill with the skill validator and inspect the final
   diff for duplicated guidance, scope drift, and accidental new authority.

### Self-update

This skill updates itself through the same process. An Encode specification can
target this section, recognition, a shared rule, or a noun definition. It must
still propose the minimum change, preserve higher-priority constraints, and
wait for an authorized `go` mode before mutation. Encoding a rule never grants
the rule, the skill, or the agent permission to publish or perform unrelated
work.
