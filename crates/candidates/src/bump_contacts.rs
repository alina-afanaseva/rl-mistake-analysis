use serde_json::Value;
use std::collections::{HashMap, HashSet};

// SubtrActor's bump detector uses the same repeat window.
const REPEAT_FRAME_WINDOW: u64 = 10;

pub(crate) struct Contact<'a> {
    pub id: &'a str,
    pub frame: u64,
    pub initiator: &'a Value,
    pub victim: &'a Value,
}

pub(crate) fn unique_contact_ids<'a>(
    contacts: impl Iterator<Item = Contact<'a>>,
) -> Result<HashSet<String>, String> {
    let mut contacts: Vec<_> = contacts.collect();
    contacts.sort_by(|left, right| left.frame.cmp(&right.frame).then(left.id.cmp(right.id)));
    let mut last_seen = HashMap::new();
    let mut identities = HashSet::new();
    let mut selected = HashSet::new();
    for contact in contacts {
        let mut pair = [contact.initiator.to_string(), contact.victim.to_string()];
        if !identities.insert((contact.frame, pair.clone())) {
            return Err("duplicate incident anchor".into());
        }
        pair.sort();
        let repeated = last_seen
            .insert(pair, contact.frame)
            .is_some_and(|previous| contact.frame - previous <= REPEAT_FRAME_WINDOW);
        if !repeated {
            selected.insert(contact.id.to_owned());
        }
    }
    Ok(selected)
}
